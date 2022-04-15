```bash
git clone https://github.com/Walstruzz/edge_eval_python.git
cd edge_eval_python/
cp -R impl ../src
cp eval_edge.py ../src
cp nms_process.py ../src
in edges_eval_plot.py:
    delete line 71-73 and 78
in edges_eval_dir.py:
    change line 10: from src.impl.bwmorph_thin import bwmorph_thin
    change line 11: from src.impl.correspond_pixels import correspond_pixels
    after line 50(for g in gt:) add:
        h, w = g.shape
        if h > w:
            g = np.flip(g,axis=1)
            g = np.transpose(g)
    after line 78(for _g in _gt:) add:
        h, w = _g.shape
            if h > w:
                _g = np.flip(_g,axis=1)
                _g = np.transpose(_g)
    delete line 105-107(from if to return) and add:
        np.savetxt(out, info, fmt="%10g")
in nms_process.py:
    add in line 5: from src.model_utils.config import config
    change line 6: from src.impl.toolbox import conv_tri, grad2
    change line 16: solver = cdll.LoadLibrary(os.path.join(config.cxx_path, "lib/solve_csa.so"))
    change line 53: assert file_format in {".mat", ".npy", ".bin"}
    after line 72(image = np.load(abs_path)) add:
        elif file_format == ".bin":
            image = np.fromfile(abs_path, dtype=np.float32).reshape(321, 481)
in eval_edge.py:
    change line 4: from src.impl.edges_eval_dir import edges_eval_dir
    change line 5: from src.impl.edges_eval_plot import edges_eval_plot
in correspond_pixels.py:
    add in line 5: from src.model_utils.config import config
    add in line 10: import os
    change line 11: solver = cdll.LoadLibrary(os.path.join(config.cxx_path, "lib/solve_csa.so"))
cp -R cxx ../src
```
