import os

def make_mad_environment(links):

    for kk in links.keys():
        os.system(f'rm {kk}')
        os.symlink(os.path.abspath(links[kk]), kk)

    # Create empty temp folder
    os.system('rm -r temp; mkdir temp')