from fastai.vision.all import *
path = untar_data(URLs.CAMVID_TINY)
path.ls()


codes = np.loadtxt(path/'codes.txt', dtype=str)
print(codes)

# # Image localization datasets
# BIWI_HEAD_POSE = f"{S3_IMAGELOC}biwi_head_pose.tgz"
# CAMVID = f'{S3_IMAGELOC}camvid.tgz'
# CAMVID_TINY = f'{URL}camvid_tiny.tgz'
# LSUN_BEDROOMS = f'{S3_IMAGE}bedroom.tgz'
# PASCAL_2007 = f'{S3_IMAGELOC}pascal_2007.tgz'
# PASCAL_2012 = f'{S3_IMAGELOC}pascal_2012.tgz'