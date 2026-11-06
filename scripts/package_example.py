"""Bundle actual CLI outputs, without absolute paths or a Python installation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def validate_outputs(output, config_id):
    """Reject stale/mixed study outputs before copying or publishing anything."""
    for name in ("analysis.json", "cad/manifest.json", "render_manifest.json"):
        path = output / name
        if not path.is_file():
            raise ValueError(f'missing output: {name}; regenerate with tesseract all')
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata.get('config_id') != config_id:
            raise ValueError(f'configuration mismatch in {name}; regenerate with tesseract all')
    rendered = json.loads((output / 'render_manifest.json').read_text(encoding='utf-8'))
    for name in ('viewer.html', 'requirements.png', 'motion.png'):
        path = output / name
        expected = rendered.get('sha256', {}).get(name)
        if not path.is_file() or expected != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'stale or changed render output: {name}; regenerate with tesseract render')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('output'))
    parser.add_argument('--destination',type=Path,default=Path('examples'))
    parser.add_argument('--config',type=Path,default=Path('config/default.toml'))
    args=parser.parse_args()
    analysis=json.loads((args.output/'analysis.json').read_text())
    manifest=json.loads((args.output/'cad/manifest.json').read_text())
    # Imported after script arguments; the documented invocation runs at repo root.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tesseract.config import load_config
    validate_outputs(args.output, load_config(args.config).digest())
    dest=args.destination; dest.mkdir(parents=True,exist_ok=True)
    required=['viewer.html','requirements.png','motion.png','analysis.json','report.md','cad/manifest.json','cad/parts.csv','cad/ASSEMBLY.md']
    for name in required:
        if not (args.output/name).is_file():
            raise ValueError(f'missing output: {name}')
    if manifest['config_id']!=analysis['config_id']:
        raise ValueError('analysis and CAD configuration IDs differ')
    for filename in ('requirements.png','motion.png','report.md'):
        shutil.copyfile(args.output/filename,dest/filename)
    shutil.copyfile(args.output/'cad/components.png',dest/'components.png')
    if (args.output/'viewer.png').exists():
        shutil.copyfile(args.output/'viewer.png',dest/'viewer.png')
    with zipfile.ZipFile(dest/'default-pack.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted(args.output.rglob('*')):
            if p.is_file():
                info=zipfile.ZipInfo(p.relative_to(args.output).as_posix(),date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,p.read_bytes())
        archive.write(args.config,'default.toml')
        archive.write('docs/manufacturing.md','manufacturing.md')
    rows=[]
    for name in ('prototype_candidate','target_candidate'):
        scan=analysis['scans'][name]; s=scan['summary']
        rows.append(f"| {s['size_mm']:g} mm | {s['edge_length_range_mm'][0]:.2f}–{s['edge_length_range_mm'][1]:.2f} | {s['min_clearance_mm']:.2f} | {s['blocked_samples']}/{s['samples']} | {sum(r['two_stage_fits'] for r in scan['telescopes'])}/32 |")
    m=analysis['config']['model']
    intervals=analysis['scans']['prototype_candidate']['summary']['clear_sample_intervals_deg']
    interval_text='、'.join(f'{a:.1f}°〜{b:.1f}°' for a,b in intervals) or 'なし'
    clear_tilts=[r['tilt_deg'] for r in manifest['socket_tilt_samples'] if r['clear']]
    blocked_tilts=[r['tilt_deg'] for r in manifest['socket_tilt_samples'] if not r['clear']]
    tilt_text=f"干渉なしの検査角度は{clear_tilts}°、干渉ありは{blocked_tilts}°"
    text=f"""# 生成例と検証結果

設定ID: `{analysis['config_id']}`。`config/default.toml`から生成した結果。

[データ・オフラインビューア・試作用CAD一式](default-pack.zip)をダウンロードし、展開した`viewer.html`をブラウザで開く。ネット接続やサーバーは不要。

![操作画面](viewer.png)

## 候補区間の結果

| 模型外形上限 | 辺長範囲 mm | 公称外形の最小すき間 mm | 干渉候補サンプル | 二段伸縮の収納・ストローク適合 |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

全周は{m['coarse_step_deg']:g}°、候補{m['candidate_start_deg']:g}°〜{m['candidate_end_deg']:g}°は{m['fine_step_deg']:g}°刻み。部品寸法は両サイズで共通。

試作サイズの公称外形で干渉がなかった連続サンプル群は{interval_text}。サンプル間の無衝突を保証しない。局所の締付けボス・ねじを含めた保守的な再検査の結果は、ZIP内の`cad/manifest.json`に記録している。

生成した二段伸縮試験片はE{manifest['sample_edge']['edge_id'] if manifest['sample_edge'] else '未割当'}に対応。球面接続を1方向で傾斜させた検査では、{tilt_text}。4辺ハブは静的な配置ゲージで、可動ハブではない。

![変形例](motion.png)
![要求寸法](requirements.png)
![試験部品](components.png)

## まだ成立していないこと

- 全32辺の二段伸縮収納、全体の連続可動、必要なジョイント自由度・可動域。
- ボスと締結部の実配置を反映した干渉解消。
- 材料・加工方式の確定、摩擦・保持力・強度・耐久性の実測。

寸法の詳細は[分析レポート](report.md)とZIP内のCSV・JSON、製作方法は[比較資料](../docs/manufacturing.md)を参照する。生成例は完成模型の製作承認や実測結果を意味しない。
"""
    (dest/'README.md').write_text(text,encoding='utf-8')
    print(f'Pack: {dest.resolve()}/default-pack.zip')


if __name__=='__main__':
    main()
