"""Install the pinned optional local LLM and CUDA 12 runtime into ignored local folders."""
from submaker.config import LocalModel
from submaker.local_install import install_local

if __name__ == "__main__":
    install_local(LocalModel(), lambda message: print(message, flush=True))
