Traceback (most recent call last):
  File "/net/module/sw/python/3.6.4/lib/python3.6/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/net/module/sw/python/3.6.4/lib/python3.6/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/__main__.py", line 88, in <module>
    optimize(vector_id_range, c, verbose=True)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/__main__.py", line 71, in optimize
    vector_id_range=vector_id_range)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/optimize.py", line 84, in optimize_multiple
    vector_id_range)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/optimize.py", line 112, in save_training_history
    np.save(seed_dir + str(iteration) + f'seed{tag}.npy', seed)
  File "<__array_function__ internals>", line 6, in save
  File "/home/pbromley/.virtualenvs/synth-seqs/lib/python3.6/site-packages/numpy/lib/npyio.py", line 541, in save
    fid = open(file, "wb")
OSError: [Errno 28] No space left on device: '/home/pbromley/projects/synth_seqs/tuning/640filters/15/seed/8394seed_0_5000.npy'
