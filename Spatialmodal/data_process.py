import ot
import os
import torch
import random
import scanpy as sc
import pandas as pd
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import anndata
import json
import h5py
from imageio import imread
from tqdm import tqdm
from scipy.sparse import csr_matrix, csc_matrix
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet152
from PIL import Image
from pathlib import Path

def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_adj=None):
    hires_path = os.path.join(file_fold, 'spatial', 'tissue_hires_image.png')
    lowres_path = os.path.join(file_fold, 'spatial', 'tissue_lowres_image.png')
    
    has_hires = os.path.exists(hires_path)
    has_lowres = os.path.exists(lowres_path)
    
    if has_hires and has_lowres:
        adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
        adata_h5 = convert_scanpy(adata_h5)
    elif has_lowres:
        adata_h5 = read_visium_lowres(file_fold, load_images=load_images, count_file=count_file)
        adata_h5 = convert_scanpy(adata_h5, use_quality='lowres')
    
    if load_images is False:
        if file_adj is None:
            file_adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")
        positions = pd.read_csv(file_adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)
        adata_h5.obsm["coord"] = adata_h5.obs.loc[:, ['array_col', 'array_row']].to_numpy()
    return adata_h5

def convert_scanpy(adata, use_quality='hires'):
    adata.var_names_make_unique()
    library_id = list(adata.uns["spatial"].keys())[0]
    if use_quality == "fulres":
        image_coor = adata.obsm["spatial"]
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + use_quality + "_scalef"
            ]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = use_quality
    # imagecolsvg and imagerowsvg only for plot svg
    adata.obs["imagecolsvg"] = adata.obsm["spatial"][:, 0]
    adata.obs["imagerowsvg"] = adata.obsm["spatial"][:, 1]
    return adata


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def get_feature(adata):
    adata_Vars = adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]
    adata.obsm['feat'] = feat


def tiling(
        adata, out_path=None, library_id: str = None, crop_size: int = 40,
        target_size: int = 299, verbose: bool = False, copy: bool = False):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Check the exist of out_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    tile_names = []

    with tqdm(
            total=len(adata),
            desc="Tiling image",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for barcode, imagerow, imagecol in zip(adata.obs.index, adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )

            tile_name = str(barcode) + str(crop_size)
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))

            tile.save(out_tile, "JPEG")
            pbar.update(1)
    # adata.obs["tile_path"] = tile_names
    tile_names = np.array(tile_names)

    out_ti_path = os.path.join(out_path, "ti_path_{}.npy".format(crop_size))
    np.save(out_ti_path, tile_names)
    return adata if copy else None

class imgDataset(Dataset):
    def __init__(self, adata, fold, img_size=40, target_size=299):
        super(imgDataset, self).__init__()
        self.obs_names = list(adata.obs.index)
        tiling_path = os.path.join(fold, f'tilingfile')
        out_ti_path = os.path.join(tiling_path, "ti_path_{}.npy".format(img_size))
        tiling(adata, tiling_path, crop_size=img_size, target_size=target_size)
        patches = []
        ti_path = np.load(out_ti_path)
        for tile_path in ti_path:
            tile = Image.open(tile_path)
            tile = np.asarray(tile, dtype="int32")
            tile = tile.astype(np.float32)
            patches.append(tile)
        patches = np.array(patches)
        adata.obsm['patches'] = patches
        self.adata = adata
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        xi = self.img_transform(self.adata.obsm['patches'][idx])
        na = self.obs_names[idx]
        return xi, na

def image_handel(xi, device, image_encoder):
    xi = xi.to(device)
    return image_encoder(xi)

def image_load(if_img, adata, img_size, fold, target_size, image_use, device):
    if image_use:
        if if_img:
            imgset = imgDataset(adata, fold, img_size=img_size, target_size=target_size)
            imgloader = DataLoader(imgset, batch_size=128, shuffle=False, pin_memory=False)
            image_encoder = resnet152(pretrained=True)
            image_encoder.requires_grad_(False)
            image_encoder.fc = nn.Identity()
            image_encoder.to(device)
            feature_dim = []
            barcode = []
            with tqdm(
                    total=len(imgloader),
                    desc="Extracting features",
                    bar_format="{l_bar}{bar} [ time left: {remaining} ]"
            ) as pbar:
                for i, (image, image_code) in enumerate(imgloader):
                    feature = image_handel(image, device, image_encoder)
                    feature_dim.append(feature.data.cpu().numpy())
                    barcode.append(image_code)
                    pbar.update(1)
            feature_dim = np.concatenate(feature_dim)
            barcode = np.concatenate(barcode)
            data_frame = pd.DataFrame(data=feature_dim, index=barcode)
            save_fileName = os.path.join(fold, 'image_feat.csv')
            data_frame.to_csv(save_fileName)
            image_representation = pd.read_csv(save_fileName, index_col='Unnamed: 0')
        else:
            save_fileName = os.path.join(fold, 'image_feat.csv')
            image_representation = pd.read_csv(save_fileName, index_col='Unnamed: 0')
        image_representation = np.array(image_representation)
    else:

        image_representation = adata.obsm['feat']
    adata.obsm['image_representation'] = image_representation


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)  # 更新为现代 API


def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n + 1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0
    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_normalized = adj_normalized.tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

###
def graph_construction(adata, n=6, dmax=50, mode='KNN'):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax)
    adj_m1 = sp.coo_matrix(adj_m1)

    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    adj_norm_m1 = preprocess_graph(adj_m1)  # 现在是 torch.sparse_coo_tensor
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])

    adj_m1 = adj_m1.tocoo()
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape).coalesce()

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1,
    }
    return graph_dict




































# def graph_construction(adata, n=6):
#     adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
#     adj_m1 = sp.coo_matrix(adj_m1)
#     adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
#     adj_m1.eliminate_zeros()
#     adj_norm_m1 = preprocess_graph(adj_m1)
#     return adj_norm_m1

def construct_interaction(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
    interaction = interaction + np.eye(n_spot)
    return interaction
def block_diag_sparse(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

    list_shapes = [a.shape for a in arrs]
    list_indices = [a.coalesce().indices().clone() for a in arrs]
    list_values = [a.coalesce().values().clone() for a in arrs]

    r_start = 0
    c_start = 0
    for i in range(len(arrs)):
        list_indices[i][0, :] += r_start
        list_indices[i][1, :] += c_start

        r_start += list_shapes[i][0]
        c_start += list_shapes[i][1]

    indices = torch.concat(list_indices, axis=1)
    values = torch.concat(list_values)
    shapes = torch.tensor(list_shapes).sum(axis=0)

    out = torch.sparse_coo_tensor(indices, values, (shapes[0], shapes[1]))

    return out


def read_visium_lowres(
    path: Path | str,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
) -> anndata.AnnData:
    path = Path(path)
    adata = sc.read_10x_h5(path / count_file, genome=genome)
    adata.uns["spatial"] = dict()

    with h5py.File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            lowres_image=path / "spatial/tissue_lowres_image.png",
        )

        adata.uns["spatial"][library_id]["images"] = dict()
        adata.uns["spatial"][library_id]["images"]["lowres"] = imread(
            str(files["lowres_image"])
        )

        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        positions = pd.read_csv(
            files["tissue_positions_file"],
            header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
            index_col=0,
        )
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")
        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        if source_image_path is not None:
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata

###



def adjacency_to_edge_index(interaction: np.ndarray) -> torch.Tensor:
    row, col = np.where(interaction.cpu() == 1)

    edge_index = torch.tensor([row, col], dtype=torch.long)

    return edge_index





def concatenate_adj_matrices(graph_dict_list):
    indices_list = []
    values_list = []
    offset_row = 0
    offset_col = 0
    total_shape = [0, 0]

    for graph_dict in graph_dict_list:
        adj_label = graph_dict["adj_label"]
        indices = adj_label.indices().numpy()
        values = adj_label.values().numpy()
        indices[0] += offset_row
        indices[1] += offset_col
        indices_list.append(indices)
        values_list.append(values)
        shape = adj_label.shape
        offset_row += shape[0]
        offset_col += shape[1]
        total_shape[0] = offset_row
        total_shape[1] = offset_col

    combined_indices = np.concatenate(indices_list, axis=1)
    combined_values = np.concatenate(values_list)
    combined_adj_label = torch.sparse_coo_tensor(
        torch.tensor(combined_indices),
        torch.tensor(combined_values, dtype=torch.float),
        total_shape
    )

    # 处理 adj_norm（torch.sparse_coo_tensor）
    norm_indices_list = []
    norm_values_list = []
    norm_offset_row = 0
    norm_offset_col = 0
    norm_total_shape = [0, 0]

    for graph_dict in graph_dict_list:
        adj_norm = graph_dict["adj_norm"].coalesce()
        indices = adj_norm.indices().numpy()
        values = adj_norm.values().numpy()
        indices[0] += norm_offset_row
        indices[1] += norm_offset_col
        norm_indices_list.append(indices)
        norm_values_list.append(values)
        shape = adj_norm.shape
        norm_offset_row += shape[0]
        norm_offset_col += shape[1]
        norm_total_shape[0] = norm_offset_row
        norm_total_shape[1] = norm_offset_col

    combined_norm_indices = np.concatenate(norm_indices_list, axis=1)
    combined_norm_values = np.concatenate(norm_values_list)
    combined_adj_norm = torch.sparse_coo_tensor(
        torch.tensor(combined_norm_indices),
        torch.tensor(combined_norm_values, dtype=torch.float),
        norm_total_shape
    )

    total_N = combined_adj_label.shape[0]
    total_E = combined_adj_label._nnz() / 2
    norm_value_block = total_N * total_N / (2 * (total_N * total_N - total_E))

    return {
        "adj_norm": combined_adj_norm,
        "adj_label": combined_adj_label,
        "norm_value": norm_value_block
    }





def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x