mkdir models
wget --no-check-certificate https://github.com/richzhang/colorization/blob/master/colorization/resources/pts_in_hull.npy?raw=true -O ./models/pts_in_hull.npy
wget --no-check-certificate https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt -O ./models/colorization_deploy_v2.prototxt
wget --no-check-certificate http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel
wget --no-check-certificate http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2_norebal.caffemodel -O ./models/colorization_release_v2_norebal.caffemodel
