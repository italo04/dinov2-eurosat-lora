import base64
import contextlib
import io
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "E2_Teardown_Intervencion_DINOv2_EuroSAT.ipynb"

def run_notebook():
    os.chdir(ROOT)
    data = json.loads(NB_PATH.read_text(encoding="utf8"))
    
    # Global execution namespace
    ns = {"__name__": "__main__", "__file__": str(NB_PATH)}
    
    # Setup dummy display function
    executed_count = 1
    
    for cell in data["cells"]:
        if cell["cell_type"] != "code":
            continue
        
        code = "".join(cell["source"])
        outputs = []
        
        # We want to capture stdout
        stdout_buf = io.StringIO()
        
        # We hook into matplotlib to capture figures
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        def custom_display(obj):
            if hasattr(obj, "to_html"):
                outputs.append({
                    "data": {
                        "text/html": [obj.to_html() + "\n"],
                        "text/plain": [str(obj) + "\n"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            else:
                outputs.append({
                    "data": {
                        "text/plain": [str(obj) + "\n"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })

        ns["display"] = custom_display
        
        original_show = plt.show
        def custom_show():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                outputs.append({
                    "data": {
                        "image/png": b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close("all")

        plt.show = custom_show

        print(f"Executing code cell {executed_count}...", flush=True)
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, ns)
        except Exception as e:
            print(f"Error in cell {executed_count}: {e}", file=sys.stderr)
            outputs.append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": [str(e)]
            })
        finally:
            plt.show = original_show

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            outputs.insert(0, {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.rstrip("\n").split("\n")]
            })

        cell["execution_count"] = executed_count
        cell["outputs"] = outputs
        executed_count += 1

    NB_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"Successfully executed and populated {executed_count-1} cells in {NB_PATH.name}!", flush=True)

if __name__ == "__main__":
    run_notebook()
