# Example of plotting radial BB separation
import matplotlib.pyplot as plt
import xtrack as xt
import numpy as np
import pandas as pd


collider = xt.Multiline.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

df_bb = {}

for beam in [1, 2]:
    bb_names = []
    bb_idx = []

    for counter, i in enumerate(collider[f"lhcb{beam}"].element_names):
        if "bb_" in i:
            bb_names.append(i)
            bb_idx.append(counter)

    radial_bb_sep = []

    for idx in bb_idx:
        element = collider[f"lhcb{beam}"].elements[idx]
        sep_x = element.other_beam_shift_x
        sep_y = element.other_beam_shift_y

        try:
            sigma_x = np.sqrt(element.other_beam_Sigma_11)
            sigma_y = np.sqrt(element.other_beam_Sigma_33)
        except:
            sigma_x = np.sqrt(element.slices_other_beam_Sigma_11[0])
            sigma_y = np.sqrt(element.slices_other_beam_Sigma_33[0])

        sep_x /= sigma_x
        sep_y /= sigma_y
        radial_bb_sep.append(np.sqrt(sep_x**2 + sep_y**2))

    df_beam = pd.DataFrame({
        "radial_bb_sep": radial_bb_sep,
        "name": bb_names
    })

    twiss = collider[f"lhcb{beam}"].twiss().to_pandas()
    df_bb[beam] = pd.merge(df_beam, twiss, on='name')


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8), sharey=True)
ax = axs.flatten()

twiss_b1 = collider[f"lhcb1"].twiss().to_pandas()
#twiss_b2 = collider[f"lhcb2"].twiss().to_pandas()

for counter, ip in enumerate([1,2,5,8]):

    plt.sca(ax[counter])

    plt.title(f"IP{ip}")

    ip_s = twiss_b1[twiss_b1.name == f"ip{ip}"]["s"].values[0]
    plt.axvline(ip_s, c='k', linestyle='--')

    df_bb_temp = df_bb[1][df_bb[1]["name"].str.contains(f".l{ip}") | df_bb[1]["name"].str.contains(f".r{ip}")]
    plt.plot(df_bb_temp.s, df_bb_temp.radial_bb_sep, marker='o', label='B1', c='b')

    #ip_s = twiss_b2[twiss_b2.name == f"ip{ip}"]["s"].values[0]
    #df_bb_temp = df_bb[2][df_bb[2]["name"].str.contains(f".l{ip}") | df_bb[2]["name"].str.contains(f".r{ip}")]
    #plt.plot(df_bb_temp.s, df_bb_temp.radial_bb_sep, marker='o', label='B2', c='r')
    
    plt.xlabel("s (m)")
    plt.ylabel("Radial BB separation  (sigma)")
    plt.grid()
    
plt.legend()
fig.tight_layout()
plt.show()
