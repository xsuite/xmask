import yaml
import re

# By default yaml interprets 2.2e11 as string and not as float
# This is a workaround to force yaml to interpret it as float
# See https://github.com/yaml/pyyaml/issues/173
# https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def load(data):
    return yaml.safe_load(data)