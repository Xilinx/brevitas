from .models import *

# Fix mnist download on older torchvision versions
from urllib import request
opener = request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
request.install_opener(opener)