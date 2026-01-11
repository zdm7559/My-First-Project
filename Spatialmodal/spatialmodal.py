from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
from numpy import linalg as LA
from sklearn.cluster import KMeans
from .data_process import *


class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(GCN_Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

# def get_proto_norm(feature, centroid, ps_label, num_protos):
#     num_data = feature.shape[0]
#     each_cluster_num = np.zeros([num_protos])

#     for i in range(num_protos):
#         each_cluster_num[i] = np.sum(ps_label == i)

#     proto_norm_term = np.zeros([num_protos])
#     for i in range(num_protos):
#         norm_sum = 0
#         for j in range(num_data):
#             if ps_label[j] == i:
#                 norm_sum = norm_sum + LA.norm(feature[j] - centroid[i], 2)
#         proto_norm_term[i] = norm_sum / (each_cluster_num[i] * np.log2(each_cluster_num[i] + 10))
#     proto_norm = torch.Tensor(proto_norm_term)
#     return proto_norm

def get_proto_loss(feature, centroid, ps_label):
    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1))
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    sim_zc = torch.matmul(feature, centroid.t())
    sim_zc_normalized = torch.exp(sim_zc)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, ps_label)
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss

class Model_Constrast(torch.nn.Module):
    def __init__(self, encoder: GCN_Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model_Constrast, self).__init__()
        self.encoder: GCN_Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    # loss definition
    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):

        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret



class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output
class InnerProductDecoder(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))

        return result
def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost+KLD

class fusion(Module):
    def __init__(self, cell_nums, dim_img_in, dim_exp_in, adj_label, device):
        super(fusion, self).__init__()
        self.device = device
        self.p_drop = 0.0
        self.mask_rate = 0.3
        self.cell_nums = cell_nums
        shared_dim_1 = 1024   
        shared_dim_2 = 512   
        shared_dim_3 = 256
        dim_exp_out_1 = 128
        gcn_hidden1 = 64
        gcn_hidden2 = 16
        dim_out = 128
        p_drop = 0.2
        dim_img_out_1 = 16
        self.adj_label = adj_label


        self.gene_pre = nn.Sequential(nn.Linear(dim_exp_in, shared_dim_1), nn.ELU())
        self.img_pre = nn.Sequential(nn.Linear(dim_img_in, shared_dim_1), nn.ELU())
        self.projector = nn.Sequential(
            nn.Linear(shared_dim_1, shared_dim_2),
            nn.ELU()
        )
        self.shared_GCN = GraphConvolution(shared_dim_2, shared_dim_3, self.p_drop)
        self.exp_GCN = GraphConvolution(shared_dim_3, dim_exp_out_1, self.p_drop)
        self.img_GCN = GraphConvolution(shared_dim_3, dim_img_out_1, self.p_drop)

        self.vgae_gcn1 = GraphConvolution(dim_exp_out_1 + dim_img_out_1, gcn_hidden1, p_drop, act=F.relu)
        self.vgae_mu = GraphConvolution(gcn_hidden1, gcn_hidden2, p_drop, act=lambda x: x)
        self.vgae_logvar = GraphConvolution(gcn_hidden1, gcn_hidden2, p_drop, act=lambda x: x)
        self.adj_mask = self.mask_generator(N=1)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)
        # 生成潜在表征
        self.fusion_linear = nn.Sequential(nn.Linear(gcn_hidden2 + dim_exp_out_1, dim_out), nn.ELU())

        self.decoder_gene = nn.Linear(dim_out, dim_exp_in)

        # constrast
        self.num_hidden_constrast = 400
        self.num_layer = 2
        self.num_proj_hidden = 256
        self.tau = 0.4
        self.activation = nn.PReLU()
        self.edge_index = adj_label.indices().to(self.device)
        self.drop_feature_rate_1 = 0.2
        self.drop_feature_rate_2 = 0.1
        self.drop_edge_rate_1 = 0.1
        self.drop_edge_rate_2 = 0.05
        self.Constrast_Encoder = GCN_Encoder(dim_exp_out_1, self.num_hidden_constrast, self.activation, k=self.num_layer)
        self.Constrast_Model = Model_Constrast(self.Constrast_Encoder, self.num_hidden_constrast, self.num_proj_hidden,
                                               self.tau)
        #cluster
        self.num_protos = 2

    def mask_generator(self, N=1):
        idx = self.adj_label.indices()

        list_non_neighbor = []
        for i in range(0, self.cell_nums):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N

            # non neighbors
            total_idx = torch.range(0, self.cell_nums-1, dtype=torch.float32).to(self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float32).to(self.device)
            random_non_neighbor = indices[:n_selected]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label.indices()[0], N)
        y = torch.concat(list_non_neighbor)

        indices = torch.stack([x, y])
        indices = torch.concat([self.adj_label.indices(), indices], axis=1)

        value = torch.concat([self.adj_label.values(), torch.zeros(len(x), dtype=torch.float32).to(self.device)])
        adj_mask = torch.sparse_coo_tensor(indices, value)

        return adj_mask

    def encoding_mask_noise(self, x, image_feat, if_mask, mask_rate=0.3):
        if if_mask:
            out_x = x.clone()
            num_nodes = x.shape[0]
            perm = torch.randperm(num_nodes, device=x.device)
            num_mask_nodes = int(mask_rate * num_nodes)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes:]
            token_nodes = mask_nodes
            out_x[token_nodes] = image_feat[token_nodes]
            return out_x, mask_nodes, keep_nodes
        else:
            return x, 'mask_nodes', 'keep_nodes'

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, feat, adj, image_feat, norm_value, if_mask):
        # mask
        feat = self.gene_pre(feat)
        image_feat = self.img_pre(image_feat)
        feat = self.projector(feat)
        image_feat = self.projector(image_feat)
        feat, mask_nodes, keep_nodes = self.encoding_mask_noise(feat, image_feat, if_mask, self.mask_rate)

        xg = self.shared_GCN(feat, adj)
        xi = self.shared_GCN(image_feat, adj)
        xg = self.exp_GCN(xg, adj)
        xi = self.img_GCN(xi, adj)
        z_gene = xg
        z_img = xi
        z_inte = torch.cat((z_gene, z_img), dim=1)

        hidden = self.vgae_gcn1(z_inte, adj)
        mu = self.vgae_mu(hidden, adj)
        logvar = self.vgae_logvar(hidden, adj)
        vgae_z = self.reparameterize(mu, logvar)

        


        loss_vgae = gcn_loss(
            preds=self.dc(vgae_z, self.adj_mask),
            labels=self.adj_mask.coalesce().values(),
            mu=mu,
            logvar=logvar,
            n_nodes=self.cell_nums,
            norm=norm_value,
        )
        
        adj_1 = dropout_adj(self.edge_index, p=self.drop_edge_rate_1)[0].to(self.device)
        adj_2 = dropout_adj(self.edge_index, p=self.drop_edge_rate_2)[0].to(self.device)
        x_1 = drop_feature(z_gene, self.drop_feature_rate_1).to(self.device)
        x_2 = drop_feature(z_gene, self.drop_feature_rate_2).to(self.device)
        z1 = self.Constrast_Model(x_1, adj_1).to(self.device)
        z2 = self.Constrast_Model(x_2, adj_2).to(self.device)
        loss_node = self.Constrast_Model.loss(z1, z2, batch_size=0)

        self.Constrast_Encoder.eval()
        z_clu = self.Constrast_Encoder(z_gene, self.edge_index).to(self.device)
        z_clu_np = z_clu.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_protos).fit(z_clu_np)
        label_kmeans = kmeans.labels_
        centers = np.array([np.mean(z_clu_np[label_kmeans == i, :], axis=0) for i in range(self.num_protos)])
        label_kmeans = label_kmeans[:, np.newaxis]
        # proto_norm = get_proto_norm(z_clu_np, centers, label_kmeans, self.num_protos)
        centers = torch.Tensor(centers).to(self.device)
        label_kmeans = torch.Tensor(label_kmeans).long().to(self.device)
        # proto_norm = torch.Tensor(proto_norm).to(self.device)
        # loss_proto = get_proto_loss(z_clu, centers, label_kmeans, proto_norm)
        loss_proto = get_proto_loss(z_clu, centers, label_kmeans)

        z_fusion = torch.cat((z_gene, vgae_z), dim=1)
        z_latent = self.fusion_linear(z_fusion)
        
        rec_gene = self.decoder_gene(z_latent)
        rec_gene_mask = self.decoder_gene(z_gene)
        
        return z_latent, rec_gene, mask_nodes, keep_nodes, rec_gene_mask, loss_node, loss_proto, loss_vgae


class SpatialModal():
    def __init__(self,
                 adata_o,
                 device=torch.device('cuda:2'),
                 learning_rate=0.001,
                 weight_decay=0.001,
                 epochs=1000,
                 random_seed=2026,
                 img_size=80,
                 target_size=299,
                 if_img=False,
                 fold="",
                 image_use=False,
                 modals='2D',
                 graph=None,
                 ):
        self.adata = adata_o.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.img_size = img_size
        self.target_size = target_size
        self.if_img = if_img
        self.fold = fold
        self.image_use = image_use
        self.cell_nums = self.adata.shape[0]
        self.modals = modals
        self.graph = graph
        fix_seed(self.random_seed)

        if 'highly_variable' not in self.adata.var.keys():
            preprocess(self.adata)
        if 'feat' not in self.adata.obsm.keys():
            get_feature(self.adata)
        if 'image_representation' not in self.adata.obsm.keys():
            image_load(self.if_img, self.adata, self.img_size, self.fold, self.target_size, self.image_use, self.device)

        if self.modals == '2D':
            graph_dict = graph_construction(self.adata, 6)
            self.adj_norm = graph_dict['adj_norm'].to(self.device)
            self.adj_label = graph_dict["adj_label"].coalesce().to(self.device)
            self.norm_value = graph_dict["norm_value"]
        else:
            self.adj_norm = self.graph['adj_norm'].to(self.device)
            self.adj_label = self.graph["adj_label"].coalesce().to(self.device)
            self.norm_value = self.graph["norm_value"]
        self.image_features = torch.FloatTensor(self.adata.obsm['image_representation'].copy()).to(self.device)
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.location = torch.FloatTensor(self.adata.obsm['spatial']).to(self.device)

    def train(self):
        self.model = fusion(self.cell_nums, self.image_features.shape[1],
                            self.features.shape[1], self.adj_label,
                            self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.learning_rate,
                                           weight_decay=self.weight_decay)
        self.model.train()
        with tqdm(
                total=self.epochs,
                desc="Training SpatialModal",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]"
        ) as pbar:
            for epoch in range(self.epochs):
                _, rec_gene, mask_nodes, keep_nodes, rec_gene_mask, loss_con, loss_proto, loss_vgae = self.model(
                    self.features,
                    self.adj_norm,
                    self.image_features,
                    self.norm_value,
                    if_mask=True
                )
                self.model.train()
                x_init_msak = self.features[mask_nodes]
                x_rec_mask = rec_gene_mask[mask_nodes]
                loss_mask = F.mse_loss(x_rec_mask, x_init_msak)
                loss_gene = F.mse_loss(rec_gene, self.features)
                if epoch < 400: 
                    gene = 1
                    con_weight = 0
                else:  
                    gene = 0.5
                    con_weight = 1

                # loss = loss_gene + 10*loss_mask + 0.01*loss_con + loss_proto
                loss = gene*(loss_gene + 10*loss_mask + loss_vgae) + con_weight*(0.01*loss_con + loss_proto)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}")
                pbar.update(1)

        with torch.no_grad():
            self.model.eval()
            self.emb_rec = self.model(self.features, self.adj_norm, self.image_features, self.norm_value,
                                      if_mask=False)[0].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
            self.adata.obsm['rec_gene'] = rec_gene.detach().cpu().numpy()
            return self.adata