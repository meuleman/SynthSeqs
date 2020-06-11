Traceback (most recent call last):
  File "/net/module/sw/python/3.6.4/lib/python3.6/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/net/module/sw/python/3.6.4/lib/python3.6/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/__main__.py", line 102, in <module>
    optimize(vector_id, target_component)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/__main__.py", line 86, in optimize
    _, _, loss, loss_vector = tuner.optimize(opt_z, target_class, iters, save_path)
  File "/home/pbromley/projects/synth_seqs/complete_run/optimize/optimize.py", line 277, in optimize
    loss = -(pred[target_class])
IndexError: index 16 is out of bounds for dimension 0 with size 16
