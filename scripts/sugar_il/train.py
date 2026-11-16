import os
import pathlib
import shutil
import sys


def _enforce_trainreg_if_present() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        return
    if sys.platform.startswith("linux") and shutil.which("trainreg") is not None:
        pid = os.getpid()
        visited = set()
        while pid > 1 and pid not in visited:
            visited.add(pid)
            try:
                cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
                comm = pathlib.Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="ignore").strip()
                if "trainreg" in comm or "trainreg" in cmdline:
                    return
                stat_text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
                pid = int(stat_text[stat_text.rfind(")") + 2 :].split()[1])
            except Exception:
                break
        sys.stderr.write(
            "\n[trainreg Compliance Error] `trainreg` is installed on this server, "
            "so direct training execution without `trainreg run --` is forbidden.\n"
            "  1) Check live balance: trainreg balance --name-id yexiaoming --json\n"
            f"  2) Launch via trainreg: trainreg run -- python {' '.join(sys.argv)}\n\n"
        )
        sys.exit(2)


if __name__ == "__main__":
    _enforce_trainreg_if_present()

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
from sugar_il.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
config_abs_path = os.path.join(os.getcwd(), 'source/sugar_il/sugar_il/config')
@hydra.main(
    version_base=None,
    # config_path=str(pathlib.Path(__file__).parent.joinpath('sugar_il','config'))
    config_path=config_abs_path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
