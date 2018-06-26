'''
Created on Jun 17, 2013
@author: WZ, justin.seeley.cn@gmail.com
'''
import numpy as np
import os, errno, itertools
import cv2
from PIL import Image
import numpy as np
import six
import logging
import math

logging.basicConfig()
lg = logging.getLogger("pongee.hammer")
lg.setLevel(logging.DEBUG)

def get_stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def dirName(path):
    '''
    Return: path if it is a dir else the dir part of the file
    '''
    return path if os.path.isdir(path) else os.path.dirname(path)


def allowedimageformats():
    '''
    Return:
        allowed images formats
    '''
    return ['bmp', 'jpg', 'png', 'BMP', 'JPG', 'PNG']


def isImage(f):
    '''
    Return: if it is an image
    '''
    return len(f) > 4 and f[-3:] in allowedimageformats()


def listImagesNonRec(basedir):
    '''
    Return: image files in this dir
    '''
    return [os.path.join(basedir, f) for f in os.listdir(basedir) if isImage(f)]


def listImagesRec(basedir):
    allfiles = []
    for d in os.listdir(basedir):
        f = os.path.join(basedir, d)
        if d not in ['.', '..'] and os.path.isdir(f):
            allfiles += listImagesRec(f)
        elif isImage(f):
            allfiles.append(f)
        else:
            pass
    return allfiles


def listImages(basedir, recursive=False):
    if not recursive:
        return listImagesNonRec(basedir)
    else:
        return listImagesRec(basedir)


def filenameFromPath(path):
    '''
    Return: base part of filename
    '''
    basepath = path.rsplit('.', 1)[0]
    rpath = basepath[::-1]
    for i in itertools.count(0):
        if rpath[i] == '/' or rpath[i] == '\\':
            break
    return rpath[0:i][::-1]


def getdirnamefrompath(path):
    basepath = path.rsplit('.', 1)[0]
    s, e = -1, -1
    rpath = basepath[::-1]
    for i in range(0, len(rpath)):
        if rpath[i] == '/' or rpath[i] == '\\':
            if s == -1:
                s = i + 1
            else:
                e = i
                break
    if e == -1:
        e = len(rpath)
    return rpath[s:e][::-1]


def allImageWithSameBase(dirname, base, getall=False):
    '''
    filter images under dirname with format 'integer.allow_image_suffix'
        * integer -> 1,2,3,...
        * allow_image_suffix -> allowedimageformats()
    Return: sequence of image names

    Raise:
        raise Exception if getall==False, and more one image with same base name
    '''
    full = ['%s.%s' % (str(base), suffix)
            for suffix in allowedimageformats()
            if os.path.exists('%s/%s.%s' % (dirname, str(base), suffix))]
    if getall:
        return full
    if len(full) > 1:
        raise Exception('more than one image with name %s' % base)

    return full[0] if len(full) == 1 else None


def suffixOfSameBaseImages(dirname, base, getall=False):
    '''
    filter images under dirname with format 'integer.allow_image_suffix'
        * integer -> 1,2,3,...
        * allow_image_suffix -> allowedimageformats()
    Return: sequence of image names

    Raise:
        raise Exception if getall==False, and more one image with same base name
    '''
    full = allImagewithbase(dirname, base, getall)
    if getall:
        return [f.rsplit('.', 1)[1] for f in full]

    return None if not full else full.rsplit('.', 1)[1]


def baseOfNumberedImages(dirname):
    '''
    filter images under dirname with format 'integer.allow_image_suffix'
        * integer -> 1,2,3,...
        * allow_image_suffix -> allowedimageformats()

    Returns: sequence of image file **BASENAME**
    '''

    return filter(lambda x: any([os.path.exists('%s/%d.%s' % (dirname, x, suffix))
                                 for suffix in allowedimageformats()]), range(0, 100))


def numberedImages(dirname):
    '''
    filter images under dirname with format 'integer.allow_image_suffix'
        * integer -> 1,2,3,...
        * allow_image_suffix -> allowedimageformats()

    Args:
        * dirname: where are images

    Return:
        * sequence of image names
    '''
    lists = baseOfNumberedImages(dirname)
    return [allImagewithbase(dirname, f) for f in lists]


def mkdir_p(*result_dirs):
    '''
    for result_dir in result_dirs:
        try:
            os.makedirs(result_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    return result_dirs


def vectorize_param(params):
    def inner(f):
        return [f(p) for p in params]

    return inner


def vectorize_f(f):
    def inner(params):
        return [f(p) for p in params]

    return inner


def splitall(path):
    """ All parts of a path

    If an absolute path, the first part is '/'; if a directory (ends with '/'), the last part is ''.

    >>> splitall('/usr/local/Cellar')
        ['/', 'usr', 'local', 'Cellar']

    >>> splitall('/usr/local/Cellar/')
        ['/', 'usr', 'local', 'Cellar', '']

    Returns: list of parts of path.
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def test_splitall():
    paths = ['/usr/local/Cellar', '/usr/local/Cellar/', 'usr/local/Cellar',
             'usr/local/Cellar/', '', '/', 'users', 'users/']
    result = [
        ['/', 'usr', 'local', 'Cellar'
        ]['/', 'usr', 'local', 'Cellar', '']['usr', 'local', 'Cellar'][
            'usr', 'local', 'Cellar', '']['']['/']['users'], ['users', '']
    ]
    for p, rp in zip(paths, result):
        assert (splitall(p) == rp)


def remove_exif(f):
    image_file = open(f)
    image = Image.open(image_file)

    # next 3 lines strip exif
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    os.remove(f)
    image_without_exif.save(f)

def findLargestContour(origin_mask):
    mask = np.copy(origin_mask)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len (contours) == 0:
        return None
    As = [cv2.contourArea(c) for c in contours]
    return contours[np.argmax(As)]


def extract_bb(origin_mask):
    c = findLargestContour(origin_mask)
    return cv2.boundingRect(c)


def expand_bb(bb, sz, marginfactor=0.2):
    margin = (bb[2] * marginfactor, bb[3] * marginfactor)

    nbb = (int(bb[0] - margin[0] / 2), int(bb[1] - margin[1] / 2),
           int(bb[2] + margin[0]), int(bb[3] + margin[1]))
    x = 0 if nbb[0] < 0 else nbb[0]
    y = 0 if nbb[1] < 0 else nbb[1]
    w = nbb[2] if nbb[0] + nbb[2] < sz[1] else sz[1] - x
    h = nbb[3] if nbb[1] + nbb[3] < sz[0] else sz[0] - y
    return (x, y, x+w, y+h)


def extract_rb(mask, factor=0.1):
    """Extract Rotated Bounding Box

    Returns: boxPoints
    """
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    R = cv2.minAreaRect(contours[0])
    return cv2.boxPoints((R[0], np.array(R[1])*(1-factor), R[2]))


def tile_ims(images, tilesize=(200, 600), tilelayout=None):
    max_w_cnt = 15
    if len(images) > max_w_cnt:
        print("support 16 windows at most")
        return

    imw, imh = tilesize
    imz = 1 if len(images[0].shape) == 2 else 3

    if tilelayout:
        iw, ih = tilelayout
    else:
        iw, ih = (len(images), 1)

    winsize = (ih * imh, iw * imw)

    im = np.zeros(winsize,
                  dtype=np.uint8) if imz == 1 else np.zeros(
                      (winsize[0], winsize[1], 3),
                      dtype=np.uint8)

    for c_m in range(0, len(images)):
        ir = c_m / iw
        ic = c_m % iw
        tim = cv2.resize(images[c_m], tilesize)
        im[ir * imh:(ir + 1) * imh, ic * imw:(ic + 1) * imw] = tim

    return im


extract_mask_bb_array = vectorize_f(extract_bb)

def expand_bb_array (bbs, sz):
    return [expand_bb(bb, sz) for bb in bbs]

def tolist(listlikearray):
    return [listlikearray[i] for i in range(listlikearray.size())]


def AABBVertexList():
    return list(itertools.product([0,1],repeat=3))

def modelAABBVertex(ext_points):
    vlist = AABBVertexList()
    return [[ext_points[r[i]][i] for i in range(3)] for r in vlist]

def modelAABBWire(model):
    box = tolist(model.aabbbox())

    m,M = np.min(np.array(box),axis=0), np.max(box, axis=0)

    edge_idx =sorted([sorted([a,b]) for (a, b) in list(itertools.permutations(list(itertools.product([0,1],repeat=3)),2))
     if sum([abs(r[0]-r[1]) for r in zip(a,b)])==1])[::2]
    return np.array([([[m,M][ia[i]][i] for i in range(3)],[[m,M][ib[i]][i] for i in range(3)]) for (ia, ib) in edge_idx])



def AABBFaceList():
    def getFaceVList(f_idx, f_pos):
        v1, v2, v3= f_idx, f_idx-1, f_idx-2
        looper = [[0,0], [0,1],[1,1], [1,0]]
        face = []
        for vv in looper:
            v = [0,]*3
            v[v1],v[v2], v[v3] = f_pos,vv[0],vv[1]
            face.append(v)
        return face

    return np.array([getFaceVList(f_idx, f_position) for f_idx in range(3) for f_position in range(2)])

def modelAABBFace(ext_points):
    flist = AABBFaceList()
    def getVertex(vlist):
        return [ext_points[vlist[i]][i] for i in range(3)]

    return np.array([[getVertex(v) for v in f] for f in flist])

import pvtrace as pv

def pointVisibility(vp_pos, points, faces3d):
    visibility = []
    for v in points:
        R = pv.Ray(vp_pos, v-vp_pos)
        rule_intersection = pv.Intersection(R, v, None)

        visible = True
        for f in faces3d:
            poly = pv.Polygon(f)
            p = poly.intersection(R)
            if p:
                I = pv.Intersection(R, p[0], None)
                if I.separation < rule_intersection.separation-1e-4:
                    visible = False
                    visibility.append(visible)
    return visibility

def edgeVisibility(vp_pos, edges, faces3d):
    if len(np.array(edges).shape) == 2:
        edges = np.array(edges).reshape(-1,2,3)
        samplepoints = [(e[0]+e[1])/2.0 for e in edges]
        visibility =  pointVisibility(vp_pos, samplepoints, faces3d)
    return visibility

def detect_lines(images, draw=False):
    """detect lines with LSD with default param."""
    d = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    lined_images = None
    detection_results = [d.detect(image)[0] for image in images]
    if draw:
        lined_images= [d.drawSegments(image, lines_std) for image, lines_std in zip(images, detection_results)]
    return detection_results, lined_images


def getCameraPosition(R, T):
    return - np.dot(np.transpose(R), T)


def toAffineRT(R, T):
    """to 4x4 rt mat"""
    rt = np.eye(4)
    rt[0:3, 0:3] = R
    rt[0:3, 3] = T
    return rt


def transformEntity(R, T, E):
    return map(lambda e: np.dot(R, e)+T, E)


def ezProject(pts, R, T, K):
    return cv2.projectPoints(np.array(pts).reshape(1,-1,3), cv2.Rodrigues(R)[0], T, K, None)[0]


def cmap(name=None, nchannel=3):
    """deprecated: to drop"""
    if name is None:
        return np.random.randint(0,255,nchannel)

    if isinstance(name, basestring):
        if name == 'side':
            return (0,0,255) if nchannel==3 else 150
        elif name == 'mid':
            return (255,0,0) if nchannel == 3 else 100
        elif name == 'line':
            return (0,255,0) if nchannel == 3 else 50
        else:
            return np.random.randint(0,255,nchannel)
    else:
        return name


#########yaml##########
import yaml

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    if len(mat.shape)>1: cols=int(mat.shape[1])
    else: cols=1
    mapping = {'rows': int(mat.shape[0]), 'cols': cols, 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)

yaml.add_representer(np.ndarray, opencv_matrix_representer)
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

def loadopencvyaml(c):
    lines = c.splitlines()
    wicked_legacy = "%YAML:1.0"

    if lines[0] == wicked_legacy:
        lines[0] = "%YAML 1.1"
    if lines[1] == '---':
        c2 = os.linesep.join(lines)
        return yaml.load(c2)
    else:
        c = "%YAML 1.1" + os.linesep + "---" + c[len(wicked_legacy):]
        return yaml.load(c)

def filename2streamwrapper(func):
    def f_(fname, *args, **kargs):
        with open(fname, 'r') as f:
            s = f.read()
            return func(s, *args, **kargs)
    return f_

loadopencvyamlfile = filename2streamwrapper(loadopencvyaml)


######Math########
def make_unit_vector(axis, ndim=3):
    avec = np.zeros(ndim)
    avec[axis] = 1.0
    return avec


def get_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """

    v1_u = get_unit_vector(v1)
    v2_u = get_unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]


def combine_rigid (outter_R, outter_T, inner_R, inner_T):
    """oR (iRX + iT) + oT = oR iR X + oR iT +oT"""
    return np.dot(outter_R, inner_R), np.dot(outter_R, inner_T.reshape(-1)) + outter_T.reshape(-1)


#########
from matplotlib import colors

def getcolor (x):
    return [int(f*255) for f in colors.ColorConverter().to_rgb(x)[::-1]]


####Patterns#####
def cached_run(f2run, rwfilenames, save_method, load_method, force = False):
    """ cache(save) to a pointed filename(s)

    rwfilenames can be a list or a string type

    For a more general:
    1. functools.lru_cache(maxsize=128, typed=False)
    2. Joblib

    """
    def cache_checker(rwfilenames):
        if isinstance(rwfilenames, six.string_types):
            cache_exist = os.path.exists(rwfilenames)
        else:
            cache_exist = reduce(lambda x,y: x and y, map(os.path.exists, rwfilenames))
        return cache_exist

    if (cache_checker(rwfilenames) is True and force is False):
        lg.debug("re-use existing: {}".format(rwfilenames))
        return load_method(rwfilenames)
    else:
        lg.debug("calculate and cache")
        ret = apply(f2run)
        save_method(rwfilenames, ret)
        return ret

def make_cached_run(save_method, load_method):
    """helper when save/load method are pointed (which is usually the case)"""
    def f(f2run, rwfilenames, force=False):
        return cached_run(f2run, rwfilenames, save_method, load_method, force)
    return f

def AddAlphaChannel(im, w=255):
    channels = cv2.split(im)
    alpha = np.ones_like(channels[0], dtype=np.uint8)*np.uint8(w)
    return cv2.merge(channels+[alpha])


def P2KRT(P):
    """decompose a projection matrix into K, R, T
    Note the T part of opencv decomposeProjectionMatrix is indeed the position of camera center in affine space (length of 4)
    """
    K, R, C = cv2.decomposeProjectionMatrix(P)[0:3]
    T = -R.dot(C[0:3]/C[-1])
    return (K, R, T.reshape(-1))

x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 1.0])

g = np.array([0.0, 0.0, -1.0])
n = np.array([0.0, 1.0, 0.0])


def logged_call(lg):
    def accept(f):
        def _F(*args, **kargs):
            ret = f(*args, **kargs)
            lg.debug("call {} with result {}".format(str(f), ret))
            return ret
        return _F
    return accept

from types import MethodType

class Proxy(object):
    def __init__(self, atarget):
        self._target = atarget

    def __getattr__(self, aname):
        atarget = self._target
        f = getattr(atarget, aname)
        if isinstance(f, MethodType):
            # Rebind the method to the atarget.
            return MethodType(f.im_func, self, atarget.__class__)
        else:
            return f

def getonlybasename(x):
    return os.path.splitext(os.path.basename(x))[0]


############ Decorators ##########
def dec_append_print(f):
    def _real_(*args):
        thelist = f(args)
        for f in thelist:
            print (f)
        return thelist

    return _real_

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def moc_provide_param(*moc_args, **moc_kargs):
    def dec_(f):
        def _f(*params, **kargs):
            # if: providing params. from un-decorated function, then use them
            # else:
            if params and kargs:
                return f(*params, **kargs)
            if params and not kargs:
                return f(*params)
            if not params and kargs:
                return f(**kargs)
            return f(*moc_args, **moc_kargs)
        return _f

    return dec_

#####loading#######

import xmltodict
def loadxmlobj(fh):
    doc = xmltodict.parse(fh)
    return doc

loadxmlobjfromfile = filename2streamwrapper(loadxmlobj)

dir_x, dir_y, dir_z = np.eye(3)


class CacheManager(object):
    @classmethod
    def get_corner_cache_file(cls, im_path):
        yaml_cache_file_name = basename_mainpart(im_path) + ".yaml"
        return yaml_cache_file_name


def find_chessboard_corners_core(board_shape, im_path, cache_dst=None, skipexisting=True):
    yaml_cache_file_name = os.path.join(cache_dst, basename_mainpart(im_path)) + ".yaml"

    if cache_dst and skipexisting:
        if os.path.exists(yaml_cache_file_name):
            return True, loadopencvyamlfile(yaml_cache_file_name)['subpixel-corners']

    im = cv2.imread(im_path, 0)
    pattern_found, corners = cv2.findChessboardCorners(
        im, board_shape, cv2.CALIB_CB_FAST_CHECK)

    if pattern_found is False:
        return pattern_found, None

    corners = corners.reshape(-1, 2)
    subcorners = np.copy(corners).astype(np.float32)

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    cv2.cornerSubPix(im, subcorners, (5, 5), (-1, -1), term)

    if cache_dst is None:
        return pattern_found, subcorners

    # cache image with corners
    im = cv2.imread(im_path)
    cv2.drawChessboardCorners(im, board_shape, subcorners, pattern_found)
    cv2.imwrite(os.path.join(cache_dst, os.path.basename(im_path)), im)

    # cache corners
    f = cv2.FileStorage(yaml_cache_file_name, cv2.FileStorage_WRITE)
    f.write("corner detection result", pattern_found)
    f.write("corners", corners)
    f.write("subpixel-corners", subcorners)
    f.release()

    return pattern_found, subcorners


class MyYamlDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyYamlDumper, self).increase_indent(flow, False)
