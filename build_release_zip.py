"""Build a local, credential-free source ZIP; does not publish or modify Git."""
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
def main():
    names=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    allowed={'.cjs','.py','.txt','.md','.css','.js','.html','.bat','.command','.sh','.pt','.onnx'}
    private={'config.json','cameras.json','plate_targets.json','vehicle_targets.json'}
    exclude={'dist','targets','vehicle_targets','alerts','uploads','experiments'}
    output=ROOT/'dist'/'Pyscript-Windows-Mac.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            path=ROOT/name
            parts=Path(name).parts
            if not name or not path.is_file() or path.is_symlink():continue
            if name in private or any(p.startswith('.') or p in exclude for p in parts):continue
            if path.suffix not in allowed and not name.endswith('.example'):continue
            archive.write(path,'Pyscript/'+name)
        assert archive.testzip() is None
    print(output)
if __name__=='__main__':main()
