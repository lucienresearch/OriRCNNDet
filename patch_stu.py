"""
X-ray patcher module.
Need kornia
"""

import numpy as np
import torch
import kornia
import os
import cv2
import random
import shutil
import math
import itertools


PARAM_IRON = [
    [0, 0, 104.3061111111111], 
    [-199.26894460884833, -1.3169138497713286, 227.17803542009827], 
    [-21.894450101465132, 0.20336113292167177, 274.63740523563814]
]

PARAM_IRON_FIX = [
    [0, 0, 104.3061111111111], 
    [0, 0, 226.1507], 
    [0, 0, 225.2509],
]


PARAM_PLASTIC = [
    [0, 0, 16.054857142857145], 
    [-175.96004580018538, -0.02797999280157535, 226.59010365257998], 
    [-1.1977592197679745, 0.03212775118421846, 251.99895369583868]
]

PARAM_GLASS = [
    [0, 0, 44.9139739229025], 
    [-162.0511635446029, -0.1537546525499077, 169.87370033743895],
    [68.9094475565913, -0.14688815701438654, 174.05450704994433]
]

MIN_EPS = 1e-6
# category_size = {
#         'glassbottle':(150,150),
#         'pressure':(150,150), 
#         'metalbottle':(150,150),
#         'umbrella':(150,150),
#         'lighter':(100,100),
#         'OCbottle':(150,150), 
#         'battery':(150,150),
#         'electronicequipment':(150, 150)
#     }

def load_infos(xlist):
    vertices = []
    faces = []

    for elm in xlist:
        try:
            if elm[0] == "v" and elm[1] != 'n':
                vertices.append([float(elm.split(" ")[1]), float(elm.split(" ")[2]), float(elm.split(" ")[3])])
            elif elm[0] == "f":
                if '//' in elm:
                    faces.append([int(elm.split(" ")[1][0]) - 1, int(elm.split(" ")[2][0]) - 1, int(elm.split(" ")[3][0]) - 1])
                else:
                    faces.append([int(elm.split(" ")[1]) - 1, int(elm.split(" ")[2]) - 1, int(elm.split(" ")[3]) - 1])
        except Exception as e:
            print('e:', str(e))
            print("load_infos err1: ", elm)
            print("load_infos err2: ", elm.split(" "))

    vertices = torch.Tensor(vertices).type(torch.cuda.FloatTensor)
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces


def load_from_file(obj, root, path, M=None):
    """
    Load vertices and faces from an .obj file.
    coordinates will be normalized to N(0.5, 0.5)
    """
    ## single part
    if type(obj) == dict:
        # name = list(obj.keys())[0]
        # path = root + name + '.obj'
        with open(path, "r") as fp:
            xlist = fp.readlines()

        vertices, faces = load_infos(xlist)
        # rotate
        if M is not None:
            vertices = torch.mm(vertices, M)   # ?????????????????????
    
        # clamp
        min_xyz, _ = torch.min(vertices, 0)
        min_xyzall = min_xyz.repeat((vertices.shape[0], 1))
        max_xyz, _ = torch.max(vertices, 0)
        max_xyzall = max_xyz.repeat((vertices.shape[0], 1))
        max_len = max_xyz - min_xyz
        max_len[-1] = 0
        max_len2, _ = torch.max(max_len, 0)
        # vertices = (vertices - min_xyzall) / max_len2
        len = max_xyzall - min_xyzall
        len[:, 0] = max_len2
        len[:, 1] = max_len2
        # vertices = (vertices - min_xyzall) / (max_xyzall - min_xyzall)
        vertices = (vertices - min_xyzall) / len
        
        faces = np.array(faces, dtype=np.int32)
        return [vertices], [faces]
    ## multi parts
    else:
        whole_name = obj[0]
        whole_path = root + whole_name + '.obj'
        with open(whole_path, "r") as fp:
            xlist = fp.readlines()
        whole_vertices, whole_faces = load_infos(xlist)

        vertices = []
        faces = []
        names = obj[-1].keys()
        for name in names:
            path = root + name + '.obj'
            with open(path, "r") as fp:
                xlist = fp.readlines()
            vertice, face = load_infos(xlist)
            vertices.append(vertice)
            faces.append(face)

        # rotate
        if M is not None:
            whole_vertices = torch.mm(whole_vertices, M)
            vertices = [torch.mm(v, M) for v in vertices]
        
        # for each v do same clamp
        min_xyz, _ = torch.min(whole_vertices, 0)
        # min_xyzall = min_xyz.repeat((whole_vertices.shape[0], 1))
        max_xyz, _ = torch.max(whole_vertices, 0)
        # max_xyzall = max_xyz.repeat((whole_vertices.shape[0], 1))
        max_len = max_xyz - min_xyz
        len = max_len.clone()
        max_len[-1] = 0
        max_len2, _ = torch.max(max_len, 0)
        len[0:2] = max_len2
        # vertices = (vertices - min_xyzall) / max_len2
        vertices_clamp = []
        for v in vertices:
            min_xyzall = min_xyz.repeat((v.shape[0], 1))
            len_all = len.repeat((v.shape[0], 1))
            
            vs = (v - min_xyzall) / len_all
            vertices_clamp.append(vs)
        
        return vertices_clamp, faces
        

def save_to_file(path, vertices, faces):
    """
    Save vertices and faces to an .obj file.
    """
    with open(path, "w") as fp:
        for i in range(vertices.shape[0]):
            fp.write("v {} {} {}\n".format(vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(faces.shape[0]):
            fp.write("f {} {} {}\n".format(faces[i][0]+1, faces[i][1]+1, faces[i][2]+1))

def get_func_hsv(params):
    def func(x, a, b, c):
        return a * torch.exp(b * x) + c

    return lambda x: torch.cat((func(x, *params[0]), func(x, *params[1]), func(x, *params[2])), 1)

def simulate(img, material="iron"):
    """
    img: Tensor (N, 1, H, W) range(0, 1) depth image
    return: (N, 3, H, W) range(0, 1) rgb image
    """
    if material == "iron":
        max_depth = 8   
        params = PARAM_IRON
    if material == "iron_fix":
        max_depth = 8
        params = PARAM_IRON_FIX
    elif material == "plastic":
        max_depth = 40
        params = PARAM_PLASTIC
    elif material == "glass":
        max_depth = 10
        params = PARAM_GLASS
    sim = get_func_hsv(params)
    img_xray = sim(img * max_depth)
    img_xray = torch.clamp(img_xray, 0, 255) / 255
    img_xray[0][0] = img_xray[0][0] * 255 * np.pi / 90
    img_xray = kornia.color.hsv_to_rgb(img_xray)
    img_xray = torch.flip(img_xray, [1])
    img_xray = torch.clamp(img_xray, 0, 1)
    img = torch.cat((img, img, img), 1)
    mask = (img!=0)
    return img_xray, mask

def get_rotate_matrix(param):
    return rotate_matrix([torch.Tensor([param[0]]), torch.Tensor([param[1]]), torch.Tensor([param[2]])])

def rotate_matrix(matrix):
    """
    Rotate vertices in a obj file.
    matrix: a three-element list [Rx, Ry, Rz], R for rotate degrees (angle system)
    """
    x = matrix[0] * np.pi / 180
    y = matrix[1] * np.pi / 180
    z = matrix[2] * np.pi / 180
    rx = torch.Tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ]).cuda()
    ry = torch.Tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ]).cuda()
    rz = torch.Tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ]).cuda()
    M = torch.mm(torch.mm(rx, ry), rz)
    return M

def is_in_triangle(point, tri_points):
    """
    Judge whether the point is in the triangle
    """
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < 1e-4:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1) & (inverDeno != 0)

def get_point_weight(point, tri_points):
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = torch.dot(v0.T, v0)
    dot01 = torch.dot(v0.T, v1)
    dot02 = torch.dot(v0.T, v2)
    dot11 = torch.dot(v1.T, v1)
    dot12 = torch.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 < 1e-4:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

def are_in_triangles(points, tri_points):
    """
    Judge whether the points are in the triangles
    assume there are n points, m triangles
    points shape: (n, 2)
    tri_points shape: (m, 3, 2)
    """
    tp = tri_points
    n = points.shape[0]
    m = tp.shape[0]

    # vectors
    # shape: (m, 2)
    v0 = tp[:, 2, :] - tp[:, 0, :]
    v1 = tp[:, 1, :] - tp[:, 0, :]
    # shape: (n, m, 2)
    v2 = points.unsqueeze(1).repeat(1, m, 1) - tp[:, 0, :]

    # dot products
    # shape: (m, 2) =sum=> (m, 1)
    dot00 = torch.mul(v0, v0).sum(dim=1)
    dot01 = torch.mul(v0, v1).sum(dim=1)
    dot11 = torch.mul(v1, v1).sum(dim=1)
    # shape: (n, m, 2) =sum=> (n, m, 1)
    dot02 = torch.mul(v2, v0).sum(dim=2)
    dot12 = torch.mul(v2, v1).sum(dim=2)

    # barycentric coordinates
    # shape: (m, 1)
    inverDeno = dot00*dot11 - dot01*dot01
    zero = torch.zeros_like(inverDeno)
    inverDeno = torch.where(inverDeno < MIN_EPS, zero, 1 / inverDeno)

    # shape: (n, m, 1)
    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno
    
    w0 = 1 - u - v
    w1 = v
    w2 = u

    # check if point in triangle
    return (u >= -MIN_EPS) & (v >= -MIN_EPS) & (u + v <= 1+MIN_EPS) & (inverDeno != 0), w0, w1, w2

def ball2depth(vertices, faces, h, w):
    """
    Save obj file as a depth image, z for depth and x,y for position
    a ball with coord in [0, 1]
    h, w: the output image height and width
    return: a depth image in shape [h, w]
    """
    vertices = torch.clamp(vertices, 0, 1)
    vs = vertices.clone()
    vs[:, 0] = vertices[:, 0] * w
    vs[:, 1] = vertices[:, 1] * h
    vertices = vs
    faces = torch.LongTensor(faces).cuda()
    
    points = torch.Tensor([(i, j) for i in range(h) for j in range(w)]).cuda()
    tri_points = vertices[faces, :2]
    in_triangle, w0, w1, w2 = are_in_triangles(points, tri_points)
    
    point_depth = w0 * vertices[faces[:, 0], 2] + w1 * vertices[faces[:, 1], 2] + w2 * vertices[faces[:, 2], 2]
    
    min_depth = torch.min(torch.where(in_triangle, point_depth, torch.full_like(point_depth, 9999)), dim=1).values
    max_depth = torch.max(torch.where(in_triangle, point_depth, torch.full_like(point_depth, -9999)), dim=1).values

    # image = torch.clamp(max_depth - min_depth, 0, 1).view(h, w)
    image = max_depth - min_depth
    image = image / image.max()
    image = torch.clamp(image, 0, 1).reshape(h, w)
    
    return image

def cal_patch_poly(patch):
    """
    calculate coordinates of four vertices of rotate bounding box
    """
    patch = patch.cpu().detach().numpy().astype(np.uint8)
    patch = patch.transpose(1, 2, 0)
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # ???????????????
    binary = np.expand_dims(binary, axis=2)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ??????????????????
    # assert len(contours) <= 5    # ????????????????????????TODO ????????????????????????
    is_pass = len(contours) <= 5   # ??????5???????????????????????????????????????patch
    print("contours:", len(contours))
    # rect = cv2.minAreaRect(contours[0])
    rect = get_max_area_rects(contours)  #???????????????????????????????????????????????????
    points = cv2.boxPoints(rect)
    points = np.int0(points)

    return points, is_pass

def get_max_area_rects(contours):
    """ ??????????????????????????? 
    (center(x,y), (width, height), angle of rotation) = cv2.minAreaRect(cont)
    """
    rects = []
    for cont in contours:
        rects.append(cv2.minAreaRect(cont))

    rects.sort(key=lambda x: x[1][0]*x[1][1], reverse=True)
    return rects[0]


def find_available_areas(img, patch_h, patch_w, img_h, img_w, border=10, topN=5):
    x_min = border
    y_min = border
    x_max = img_h - patch_h - border
    y_max = img_w - patch_w - border
    assert x_max > x_min and y_max > y_min, "img_w:{}, img_h:{}, patch_w:{}, patch_h:{}".format(img_w, img_h, patch_w, patch_h)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array((0, 20, 50), np.uint8)
    high_hsv = np.array((100, 255, 255), np.uint8)
    mask = cv2.inRange(hsv, low_hsv, high_hsv)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # count and filter all available areas
    areas = []
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (area >= 100) and (area <= img_h*img_h/4) and (x >= x_min and x <= x_max and y >= y_min and y <= y_max):
            dist = math.sqrt((x-img_w/2)*(x-img_w/2) + (y-img_h/2)*(y-img_h/2))
            areas.append([contour, dist, area, x, y])
    # sort areas
    areas = sorted(areas, key=lambda x: x[2], reverse=1)
    areas = list(itertools.islice(areas, 30))

    areas = sorted(areas, key=lambda x: -x[1]*2+x[2]*0.4, reverse=1)
    topNAreas = list(itertools.islice(areas, topN))
    fallback = (random.randint(x_min, x_max), random.randint(y_min, y_max))
    return (topNAreas, fallback)

def find_stick_point(available_areas, used_areas):
    """
    Find stick point randomly
    """
    if len(available_areas) == 0 or len(available_areas) == len(used_areas):
        return None
    while True:
        randint = random.randint(0, len(available_areas)-1)
        if randint not in used_areas:
            area = available_areas[random.randint(0, len(available_areas)-1)]
            used_areas.append(randint)
            return (area[3], area[4])

def save_img(path, img_tensor, shape):
    img_tensor = img_tensor.cpu().detach().numpy().astype(np.uint8)
    img = img_tensor.transpose(1, 2, 0)
    img = cv2.resize(img, (shape[1], shape[0]))
    cv2.imwrite(path, img)


def random_samples_objs(obj_root, index):
    """ ??? obj ?????????????????? """

    # ???????????????????????????????????? TODO iron_fix ??????????????????  umbrella ??????????????????????????????????????????
    material_dic = {
                    'glassbottle':'glass',
                    'pressure':'iron', 
                    'metalbottle':'iron',
                    'umbrella':'plastic', 
                    'lighter':'plastic', 
                    'OCbottle':'plastic', 
                    'battery':'plastic',
                    'electronicequipment':'plastic',
                    }              

    # ??????????????????????????????
    categorys = list(material_dic.keys())
    rand_cat_idx = random.randint(0, len(categorys)-1)
    # rand_cat_idx = index % len(categorys)
    category = categorys[rand_cat_idx]

    # ????????????????????????????????? obj
    obj_names = [obj_file for obj_file in os.listdir(os.path.join(obj_root, category)) if obj_file.endswith('.obj')]
    rand_obj_idx = random.randint(0, len(obj_names)-1)
    # rand_obj_idx = index % len(obj_names)
    obj_path = os.path.join(obj_root, category, obj_names[rand_obj_idx])
    obj_name = obj_names[rand_obj_idx].split('.')[0]

    random_size = random.randint(100, 150) # ????????????patch??????
        
    obj_dict = {                                     # .obj file name and infos
        f'{category}_{obj_name}_vf': {
                'category': category, 
                'multi-part': False,
                'material':material_dic[category],
                'patch_size': (random_size, random_size)      
            }
    }   

    print('obj_path:', obj_path)

    return obj_dict, obj_path

def save_patch(patch):
    cv2.imwrite

rotate_param_dict = {
    'lighter/2.obj': [
        [200, 200, 100],
        [200, 200, 200],
        [200, 200, 90],
        [1, 1, 1],
        [1, 2, 1],
        [5, 1, 3]
    ],
    'lighter/3.obj': [
        [100, 100, 200],
        [100, 100, 300],
        [100, 100, 250],
        [200, 100, 250],
        [200, 150, 250],
        [200, 150, 100]
    ],
    'lighter/4.obj': [
        [200, 100, 200],
        [1, 100, 1],
        [300, 300, 200],
        [1, 100, 100],
        [100, 100, 200],
        [200, 150, 250]
    ],
    'lighter/5.obj': [
        [100, 100, 50],
        [200, 100, 200],
        [300, 300, 200],
        [200, 150, 100]
    ],
    'lighter/7.obj': [
        [100, 100, 100],
        [100, 100, 200],
        [100, 100, 300],
        [200, 100, 300],
        [200, 300, 300],
        [300, 300, 300]
    ]
}


def get_rotate_param_old(obj_path):
    # ??????window?????????????????????
    obj_path = obj_path.replace("\\", "/")
    path_split = obj_path.split("/")
    category = path_split[len(path_split) - 2]
    filename = path_split[len(path_split) - 1]
    key = category + "/" + filename
    param_list = rotate_param_dict.get(key,[])
    param = [random.randint(50, 250) for _ in range(3)]
    index = random.randint(0, len(param_list))
    # ???????????????????????????????????????
    if param_list is not None and index < len(param_list):
        param = param_list[index]
    return param


def get_rotate_param(obj_path):
    # ??????window?????????????????????
    obj_path = obj_path.replace("\\", "/")
    path_split = obj_path.split("/")
    category = path_split[len(path_split) - 2]
    filename = path_split[len(path_split) - 1]
    x = random.randint(0, 360)
    y = random.randint(0, 360)
    z = random.randint(0, 360)

    if category in ['metalbottle', 'glassbottle']:
        while (60 <= x <= 120 or 240 <= x <= 300) and (0 <= y <= 20 or 160 <= y <= 200 or 340 <= y <= 360):
            x = random.randint(0, 360)
            y = random.randint(0, 360)

    if category == 'lighter':
        key = category + "/" + filename
        param_list = rotate_param_dict.get(key, [])
        index = random.randint(0, len(param_list))
        # ???????????????????????????????????????
        if param_list is not None and index < len(param_list):
            x, y, z = param_list[index]
            
    return [x, y, z]


def gen_patch(obj_dict, obj_path):
    """ ?????? obj ?????? patch """

    # ??????????????????
    # rotate_param = [110, 38, 56]                             # rotate config (x, y, z)
    # rotate_param = [random.randint(50, 250) for _ in range(3)]   # ????????????????????????
    rotate_param = get_rotate_param(obj_path)
    M = get_rotate_matrix(rotate_param)

    obj_root = os.path.dirname(obj_path)
    vertices, faces = load_from_file(obj_dict, obj_root, obj_path, M) # load obj vertices and faces with rotation

    obj_dict_info = obj_dict

    print('------------Part (2): Generate pacthed image.------------')

    print('------------Step (1): Generate depth image.------------')
    # we only consider object has one part 
    v = vertices[0]
    f = faces[0]
    name = list(obj_dict_info.keys())[0]
    obj_infos = list(obj_dict_info.values())[0]

    # patch_size = category_size[obj_infos['category']] 
    patch_size = obj_infos['patch_size']
    group = torch.clamp(v, 0, 1)
    depth_clamp = ball2depth(group, f, patch_size[0], patch_size[1]).unsqueeze(0)

    # # Enlarge the pixel to the normal image range
    # depth_img = depth_clamp[:,:] * 255
    # # print(depth_img.shape)
    # save_img(depth_save_path+'/' + name + '_depth.png',depth_img, patch_size) 

    print('------------------Step (2): Generate patch.------------------')
    material = obj_infos['material']
    patch, mask = simulate(depth_clamp.unsqueeze(0), material)
    patch[~mask] = 1

    # # Enlarge the pixel to the normal image range
    # patch_img = patch.squeeze(0) * 255
    # save_img(patch_save_path+'/' + name + '_patch.png',patch_img, patch_size)


    return patch, mask, rotate_param


def parse_patches(data_root, obj_root, mode='train', num=-1):
   
    # ???????????????
    img_root = data_root + f'{mode}/images/'
    patched_img_root = data_root + f'{mode}/images_patched/'
    if os.path.exists(patched_img_root):
        shutil.rmtree(patched_img_root)
    os.mkdir(patched_img_root)

    # ???????????????
    ann_root = data_root + f'{mode}/annotations'
    patched_ann_root = data_root + f'{mode}/annotations_patched/'
    if os.path.exists(patched_ann_root):
       shutil.rmtree(patched_ann_root) 
    os.mkdir(patched_ann_root)

    # ?????????????????????
    img_save_path = data_root + f'{mode}/results/'
    if os.path.exists(img_save_path):
        shutil.rmtree(img_save_path)
    os.mkdir(img_save_path)

    # ?????????????????????
    eval_ann_root = data_root +f'{mode}/annotations_eval/'
    if os.path.exists(eval_ann_root):
        shutil.rmtree(eval_ann_root)
    os.mkdir(eval_ann_root)
        

    # ??????????????????
    img_names = [name for name in os.listdir(img_root) if name.endswith('.jpg')]
    img_names.sort()
    num = num if num > 0 else len(img_names)

    cc = 0
    patch_total = 0
    for i, img_name in enumerate(img_names[:num]):
    
        img_id = img_name.split('.')[0]  # img_id = 'train00001'

        ## Step (4): stick patch to image
        print('------------------Step (3): Stick patch.------------------')     
        ## load img
        print(img_id)
        img_path = os.path.join(img_root, img_id+'.jpg')
        img = cv2.imread(img_path)
        img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)  #lzk ?????????????????????
        img_tensor = img_tensor.cuda()

        ann_path = os.path.join(ann_root, img_id+'.txt')
        anns = open(ann_path, 'r').readlines()
        eval_anns = []
        points_all = []

        used_areas = []
        count = 0
        # patch_count = random.randint(2,3)
        while count < 3: # ?????????????????? 3 ??? patch

            obj_dict, obj_path = random_samples_objs(obj_root, i)
            patch, mask, rotate_param = gen_patch(obj_dict, obj_path)

            # we only consider object has one part 
            name = list(obj_dict.keys())[0]
            obj_infos = list(obj_dict.values())[0]

            # patch_size = category_size[obj_infos['category']] 
            patch_size = obj_infos['patch_size']
            
            img_h, img_w = img_tensor.shape[1:]
            available_areas, fallback = find_available_areas(img, patch_size[0], patch_size[1], img_h, img_w, topN=1)
            point = find_stick_point(available_areas, used_areas)  # ????????????????????????
            if point is None:
                print('Using fallback')
                point = fallback
                cc += 1

            # calculate coordinates of four vertices of rotate bounding box
            patch[~mask] = 0
            points, is_pass = cal_patch_poly(patch.squeeze(0) * 255)
            # ?????? patch ????????????????????????
            if not is_pass:
                continue
            else:
                count+=1

            patch_total += 1

             # stick patch
            patch[~mask] = 1
            img_tensor[:, point[0]:point[0]+patch_size[0], point[1]:point[1]+patch_size[1]].mul_(patch.squeeze(0))
            
            # add stick point offset
            points[:, 0] += point[1]
            points[:, 1] += point[0]
            
            # make annotation format 
            points_list = points.reshape(-1).tolist()
            points_str = [str(i) for i in points_list]
            category = obj_infos['category']
            new_ann = [img_id+'.jpg', '1', category, '0 0 0 0']
            new_ann = new_ann + points_str
            str_new_ann = ' '.join(new_ann) + '\n'
            anns.append(str_new_ann)
            points_all.append(points)


            ## Step (6): save eval annotation
            eval_ann = []
            eval_ann.append(category)
            eval_ann.append(name)
            eval_ann.extend([str(i) for i in rotate_param])
            eval_ann.extend([str(i) for i in point])
            eval_ann.extend([str(i) for i in patch_size])
            eval_anns.append(eval_ann)

        print("fallback count:{}, total patch count:{}".format(cc, patch_total))

        img_id = img_id +'_syn'
        new_img_path = os.path.join(patched_img_root, img_id+'.jpg')
        save_img(new_img_path, img_tensor, img_tensor.shape[1:])

        # ?????????????????????????????????????????????
        img = cv2.imread(new_img_path)
        for points in points_all:
            img = cv2.drawContours(img, [points], 0, (0, 0, 255), 2)
        save_img_name = img_id.split('.')[0] + '_rec_patch.jpg'
        cv2.imwrite(img_save_path + '/' + save_img_name, img)

        print('------------------part (3): Save new annotation.------------------') 
        # write to file
        new_ann_path = os.path.join(patched_ann_root, img_id+'.txt')
        new_anns_file = open(new_ann_path, 'w')
        for ann in anns:
            new_anns_file.write(ann)
        new_anns_file.close()

        print('------------------Part (4): Save eval annotation.------------------') 
        eval_ann_path = os.path.join(eval_ann_root, img_id+'.txt')
        with open(eval_ann_path, 'w') as f:
            for ann in eval_anns:
                f.write(' '.join(ann) + '\n')

if __name__ == '__main__':
    ## config
    global current_path
    current_path = os.getcwd()
    data_root = f'{current_path}/mmrotate/data/datasets_hw/'
    obj_par = f'{current_path}/mmrotate/data/'
    obj_root = obj_par + 'objs/'                                    # obj root path
    # depth_save_path = obj_par + 'depthes'                           # depth save path
    # patch_save_path = obj_par + 'patches'                           # patch save path

    parse_patches(data_root, obj_root, mode='train')