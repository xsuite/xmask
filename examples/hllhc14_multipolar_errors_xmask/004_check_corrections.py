import xtrack as xt
import matplotlib.pyplot as plt

env_test = xt.load('lhc_arc_errors_with_correction.json')
env_ref = xt.load('../hllhc14_multipolar_errors_legacy/'
                   'collider_errors_on_corrections_on.json')

line_name = 'lhcb1'

line_test = env_test[line_name]
line_ref = env_ref[line_name]

tt_ref = line_ref.get_table(attr=True)
tt_test = line_test.get_table(attr=True)

# kill IR non-linearities in reference machine (for comparison)
for ipn in [1, 2, 5, 8]:
    start_ir = f'e.ds.l{ipn}.b1'
    end_ir = f's.ds.r{ipn}.b1'
    tt = line_ref.get_table()
    tt_ir = tt.rows[start_ir:end_ir]
    for nn in tt_ir.name:
        if hasattr(line_ref[nn], 'knl'):
            line_ref[nn].knl[2:] = 0
        if hasattr(line_ref[nn], 'ksl'):
            line_ref[nn].ksl[1:] = 0

tw_test = line_test.twiss4d()
tw_ref = line_ref.twiss4d()

twom_test = line_test.twiss4d(coupling_edw_teng=True, delta0=0.5e-3)
twom_ref = line_ref.twiss4d(coupling_edw_teng=True, delta0=0.5e-3)

nlchr_test = line_test.get_non_linear_chromaticity(num_delta=20)
nlchr_ref = line_ref.get_non_linear_chromaticity(num_delta=20)

tt_on_corr = env_test.vars.get_table().rows['on_corr_k2sl.*']
assert len(tt_on_corr) == 9
env_test.set(tt_on_corr.name, 0)
nlchr_test_no_corr = line_test.get_non_linear_chromaticity(num_delta=20)


plt.close('all')
plt.figure(1)
plt.plot(1e3 * nlchr_test_no_corr.delta0, [ttt.c_minus for ttt in nlchr_test_no_corr.twiss], label='Test no correction')
plt.plot(1e3 * nlchr_test.delta0, [ttt.c_minus for ttt in nlchr_test.twiss], label='Test')
plt.plot(1e3 * nlchr_ref.delta0, [ttt.c_minus for ttt in nlchr_ref.twiss], label='Reference', linestyle='dashed')
plt.xlabel(r'$\delta_0$ [10$^{-3}$]')
plt.ylabel('C-')
plt.legend()
plt.suptitle(line_name)
plt.show()