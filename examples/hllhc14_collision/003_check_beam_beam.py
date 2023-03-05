# %%
import yaml
import json
import xobjects as xo
import xtrack as xt
import xfields as xf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# %% Import the weak and strong lines
verbose = True
with open('collider_02_bb_on.json', 'r') as fid:
    collider = xt.Multiline.from_dict(json.load(fid))

collider.build_trackers()

# %%
weak_beam = 'lhcb1'
strong_beam = 'lhcb2'
ip_name = 'ip1'

beams = {'weak':{'name':weak_beam},'strong':{'name':strong_beam}}

## BB ON
collider.vars['beambeam_scale'] = 1
if verbose: print(f"=> BB ON")

for beam in beams:
    my_beam =  beams[beam]
    if verbose: print(f"==> {beam} beam: { my_beam['name']}")
    my_beam['line'] = collider[my_beam['name']]
    if beam == 'weak':
        my_beam['twiss_bb_on'] = my_beam['line'].twiss()
        my_beam['survey_bb_on'] = my_beam['line'].survey(element0=ip_name)
    elif beam == 'strong':
        my_beam['twiss_bb_on'] = my_beam['line'].twiss(reverse=True)
        my_beam['survey_bb_on'] = my_beam['line'].survey(element0=ip_name, reverse=True)
    else:
        raise Exception("Only two beams are possible (the weak and the strong).")

## BB OFF
if verbose: print(f"=> BB OFF")
collider.vars['beambeam_scale'] = 0
for beam in beams:
    my_beam =  beams[beam]
    if verbose: print(f"==> {beam} beam: { my_beam['name']}")
    if beam == 'weak':
        my_beam['twiss_bb_off'] = my_beam['line'].twiss()
        my_beam['survey_bb_off'] = my_beam['line'].survey(element0=ip_name)
    elif beam == 'strong':
        my_beam['twiss_bb_off'] = my_beam['line'].twiss(reverse=True)
        my_beam['survey_bb_off'] = my_beam['line'].survey(element0=ip_name, reverse=True)
    else:
        raise Exception("Only two beams are possible (the weak and the strong).")

## BB BACK ON
collider.vars['beambeam_scale'] = 1

# %% Test on CO
print('Testing the effect of the BB on the CO: BB should have no effect on the CO.')
for beam in beams:
    my_beam =  beams[beam]
    if verbose: print(f"=> {beam} beam: { my_beam['name']}")
    if verbose: print(f"==> twiss values")
    for ii in ['x', 'y', 'delta']:
        np.testing.assert_allclose(my_beam['twiss_bb_on'][ii], my_beam['twiss_bb_off'][ii],  atol=1e-10, rtol=0)
        if verbose: print(f'{ii} is OK')
    for ii in ['px', 'py']:
        np.testing.assert_allclose(my_beam['twiss_bb_on'][ii], my_beam['twiss_bb_off'][ii],  atol=1e-12, rtol=0)
        if verbose:print(f'{ii} is OK')
    for ii in ['zeta']:
        np.testing.assert_allclose(my_beam['twiss_bb_on'][ii], my_beam['twiss_bb_off'][ii],  atol=1e-9, rtol=0)
        if verbose: print(f'{ii} is OK')
    for ii in ['delta']:
        np.testing.assert_allclose(my_beam['twiss_bb_on'][ii], my_beam['twiss_bb_off'][ii],  atol=1e-10, rtol=0)
        if verbose: print(f'{ii} is OK')
    if verbose: print(f"==> survey values")
    for ii in ['X','Y','Z','theta','phi','psi','angle','tilt']:
            np.testing.assert_allclose(my_beam['survey_bb_on'][ii], my_beam['survey_bb_off'][ii],  atol=1e-200, rtol=0)
            if verbose: print(f'{ii} is OK')

# %%
def get_df_elements_of_type(self, my_type):
    aux = self.get_elements_of_type(my_type)
    aux = aux + (self.get_s_position(at_elements=aux[1]),)
    return [{'name':jj, 's':kk, 'element':ii} for ii,jj,kk in zip(aux[0],aux[1],aux[2])]
xt.line.Line.get_df_elements_of_type=get_df_elements_of_type

for beam in beams:
        my_beam =  beams[beam]
        my_beam['bb_2d_dict'] =my_beam['line'].get_df_elements_of_type(xf.beam_elements.beambeam2d.BeamBeamBiGaussian2D)
        my_beam['bb_3d_dict'] =my_beam['line'].get_df_elements_of_type(xf.beam_elements.beambeam3d.BeamBeamBiGaussian3D)
# %% making the dictionary of the BB elements
ho_slices = 11
ip_list = [1,2,5,8]
bblr_lenses_installed = [25,20,25,20]
harmonic = 3564
for beam in beams:
    my_beam =  beams[beam]
    my_beam['bb_2d'] = {}
    for ip in ip_list:
            my_beam['bb_2d'][ip] = {'left':[], 'right':[]}
            for bblr in my_beam['bb_2d_dict']:
                if verbose: print(bblr['name'])
                if f'l{ip}' in bblr['name']:
                    my_beam['bb_2d'][ip]['left'].append(bblr)
                if f'r{ip}' in bblr['name']:
                    my_beam['bb_2d'][ip]['right'].append(bblr)
                assert bblr['name'][0:5]=='bb_lr'
    my_beam['bb_3d'] = {}
    for ip in ip_list:
            my_beam['bb_3d'][ip] = {'left':[], 'right': [], 'center': []}
            for bbho in my_beam['bb_3d_dict']:
                if verbose: print(bbho['name'])
                if f'l{ip}' in bbho['name']:
                    my_beam['bb_3d'][ip]['left'].append(bbho)
                if f'r{ip}' in bbho['name']:
                    my_beam['bb_3d'][ip]['right'].append(bbho)
                if f'c{ip}' in bbho['name']:
                    my_beam['bb_3d'][ip]['center'].append(bbho)
                assert bbho['name'][0:5]=='bb_ho'

# %% Checking the numbers of the BB encounters
for beam in beams:
    my_beam =  beams[beam]
    for ip, bblr_number in zip(ip_list, bblr_lenses_installed):
        # check the BBLR 
        assert len(my_beam['bb_2d'][ip]['left'])==len(my_beam['bb_2d'][ip]['right'])==bblr_number
        # check the BBHO 
        assert len(my_beam['bb_3d'][ip]['left'])==len(my_beam['bb_3d'][ip]['right'])==(ho_slices-1)/2
        assert len(my_beam['bb_3d'][ip]['center'])==1
        
# %% Check the position of the BB encounters

from scipy.stats import norm
sigmaz=0.076
# assert that ho_slices is odd
assert ho_slices%2==1
assert beams['weak']['line'].get_length()==beams['strong']['line'].get_length()

# from  https://github.com/giadarol/WeakStrong/blob/master/slicing.py
def get_z_centroids(ho_slices, sigmaz):
    z_cuts = norm.ppf(np.linspace(0,1,ho_slices+1)[1:int((ho_slices+1)/2)])* sigmaz
    z_centroids = []
    z_centroids.append(-sigmaz/np.sqrt(2*np.pi)*np.exp(-z_cuts[0]**2/(2*sigmaz*sigmaz))*float(ho_slices))
    for ii,jj in zip(z_cuts[0:-1],z_cuts[1:]):
        z_centroids.append(-sigmaz/np.sqrt(2*np.pi)*
            (np.exp(-jj**2/(2*sigmaz*sigmaz))-
                np.exp(-ii**2/(2*sigmaz*sigmaz))
            )*ho_slices)
    return z_centroids +[0] + [-ii for ii in z_centroids[-1::-1]]

bblr_distance = beams['weak']['line'].get_length()/harmonic/2
atol = 1e-12
# Please note that the Left and Right are set with respect to the lhcb1
for beam in beams:
    my_beam =  beams[beam]
    for ip in ip_list:
        if  my_beam['name']=='lhcb1':
            for ii, bblr in enumerate(my_beam['bb_2d'][ip]['left'][-1::-1]):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                np.testing.assert_allclose(
                    (bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'))/bblr_distance,-(ii+1),
                     atol=atol, rtol=0)
            for ii, bblr in enumerate(my_beam['bb_2d'][ip]['right']):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                np.testing.assert_allclose(
                    (bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'))/bblr_distance,(ii+1),
                     atol=atol, rtol=0)
            for ii, bblr in enumerate(my_beam['bb_3d'][ip]['right'][-1::-1]):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                # TODO: check the position of the BBHO
                np.testing.assert_allclose(bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'),
                                            -get_z_centroids(ho_slices, sigmaz)[ii]/2)
            for ii, bblr in enumerate(my_beam['bb_3d'][ip]['left']):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                # TODO: check the position of the BBHO
                np.testing.assert_allclose(bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'),
                                            get_z_centroids(ho_slices, sigmaz)[ii]/2)
             
        elif my_beam['name']=='lhcb2':
            for ii, bblr in enumerate(my_beam['bb_2d'][ip]['left']):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                np.testing.assert_allclose(
                    (bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'))/bblr_distance,+(ii+1),
                     atol=atol, rtol=0)
            for ii, bblr in enumerate(my_beam['bb_2d'][ip]['right'][-1::-1]):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                np.testing.assert_allclose(
                    (bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'))/bblr_distance,-(ii+1),
                     atol=atol, rtol=0)
            for ii, bblr in enumerate(my_beam['bb_3d'][ip]['right']):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                # TODO: check the position of the BBHO
                np.testing.assert_allclose(bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'),
                                            get_z_centroids(ho_slices, sigmaz)[ii]/2)
            for ii, bblr in enumerate(my_beam['bb_3d'][ip]['left'] [-1::-1]):
                if verbose: print(f'{beam} beam, '+ bblr['name'])
                # TODO: check the position of the BBHO
                np.testing.assert_allclose(bblr['s']-my_beam['line'].get_s_position(at_elements=f'ip{ip}'),
                                            -get_z_centroids(ho_slices, sigmaz)[ii]/2)
        else:
            raise Exception("Only two beams are possible (the weak and the strong).")
        assert len(my_beam['bb_3d'][ip]['center'])==1
        assert my_beam['bb_3d'][ip]['center'][0]['s']==my_beam['line'].get_s_position(at_elements=f'ip{ip}')


# %% Check the beam size at the BB encounters
def change_bblr_weak_strong(my_string):
    if 'b1' in my_string:
        return my_string.replace('b1','b2')
    elif 'b2' in my_string:
        return my_string.replace('b2','b1')
    else:
        raise Exception("Please check the name of the BB element")
    
epsilon_xn = 2.0e-6 # m
epsilon_yn = 3.0e-6 # m

for beam, other_beam in zip([beams['weak'], beams['strong']],[beams['strong'], beams['weak']]):
    beam_twiss_df = beam['twiss_bb_off'].to_pandas()
    other_beam_twiss_df = other_beam['twiss_bb_off'].to_pandas()
    beam_survey_df = beam['survey_bb_off'].to_pandas()
    other_beam_survey_df = other_beam['survey_bb_off'].to_pandas()

    for ip in beam['bb_2d']:
        for side in beam['bb_2d'][ip]:
            for ii in beam['bb_2d'][ip][side]:
                my_dict =  ii['element'].to_dict()
                print(ii['name'])
                assert my_dict['scale_strength']==1
                assert my_dict['other_beam_q0']==1
                assert my_dict['other_beam_num_particles']==2.2e11
                assert other_beam['twiss_bb_off'].particle_on_co.beta0[0] == my_dict['other_beam_beta0']
                betx=other_beam_twiss_df[other_beam_twiss_df['name']==
                                         change_bblr_weak_strong(ii['name'])].betx1.values[0]
                bety=other_beam_twiss_df[other_beam_twiss_df['name']==
                                         change_bblr_weak_strong(ii['name'])].bety2.values[0]
                sigma11 = (betx*epsilon_xn
                           /other_beam['twiss_bb_off'].particle_on_co.beta0[0]
                           /other_beam['twiss_bb_off'].particle_on_co.gamma0[0])
                sigma33 = (bety*epsilon_yn
                           /other_beam['twiss_bb_off'].particle_on_co.beta0[0]
                           /other_beam['twiss_bb_off'].particle_on_co.gamma0[0])
                assert my_dict['other_beam_Sigma_13'] == 0 
                np.testing.assert_allclose(my_dict['other_beam_Sigma_33'], sigma33)
                assert my_dict['min_sigma_diff'] == 1e-10
                #print(-beam_twiss_df[beam_twiss_df['name']==ii['name']].y.values[0]- my_dict['ref_shift_y'])
# %%
