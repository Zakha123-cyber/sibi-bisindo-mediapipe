"""Report completed, duplicate-filtered recovery without opening test data."""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'data/features/bisindo-recovery-filtered-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'data/reports/bisindo-recovery')
    args=parser.parse_args()
    source=args.cache
    report=json.loads((source/'summary.json').read_text())
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key,stats in report['splits'].items():
        split=key.split('/')[1]
        for label,counts in stats['per_class'].items():
            before=stats['baseline']['per_class'][label]
            rows.append(dict(split=split,label=label,total=counts['total'],valid_before=before['feature_valid'],valid_after=counts['feature_valid'],
                             rate_before=100*before['feature_valid']/counts['total'],rate_after=100*counts['feature_valid']/counts['total'],
                             two_before=before['hands'].get('2',0),two_after=counts['hands'].get('2',0)))
    with (output/'per_class.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    splits=[s for s in ['train','val'] if 'bisindo/'+s in report['splits']]
    for column,split in enumerate(splits):
        stats=report['splits']['bisindo/'+split];baseline=stats['baseline']
        before=baseline['feature_valid_rate']*100;after=stats['feature_valid_rate']*100
        axes[0].bar(column-.18,before,width=.36,color='#718096',label='Baseline' if column==0 else None)
        axes[0].bar(column+.18,after,width=.36,color='#169b78',label='Recovery + filter' if column==0 else None)
        axes[0].text(column-.18,before+1,f'{before:.2f}%',ha='center');axes[0].text(column+.18,after+1,f'{after:.2f}%',ha='center')
        for index,stats_kind in enumerate([baseline,stats]):
            offset=0
            for hands,color in zip(['0','1','2'],['#d66b39','#5ca4cd','#2b6cb0']):
                pct=100*stats_kind['hand_counts'].get(hands,0)/stats_kind['total']
                x=column*3+index
                axes[1].bar(x,pct,bottom=offset,color=color,label=hands+' tangan' if column==0 and index==0 else None)
                if pct>6:axes[1].text(x,offset+pct/2,f'{pct:.1f}%',ha='center',va='center',fontsize=9)
                offset+=pct
    axes[0].set_xticks(range(len(splits)),[s.upper() for s in splits]);axes[0].set_ylim(0,110)
    axes[0].set_ylabel('% gambar dengan fitur valid');axes[0].set_title('Cakupan sebelum / sesudah');axes[0].legend(loc='lower right')
    axes[1].set_xticks([i*3+j for i in range(len(splits)) for j in range(2)],
                       [f'{s}\n{state}' for s in splits for state in ['lama','baru']]);axes[1].set_ylim(0,105)
    axes[1].set_title('Distribusi jumlah tangan terdeteksi');axes[1].legend(loc='lower right')
    fig.suptitle('Pemulihan landmark BISINDO — bukan akurasi classifier atau pengesahan pose',fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/'overview.png',dpi=160);plt.close(fig)
    labels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'); matrix=np.full((26,len(splits)*2),np.nan)
    for row in rows:
        j=splits.index(row['split'])*2;i=labels.index(row['label'])
        matrix[i,j:j+2]=[row['rate_before'],row['rate_after']]
    fig,axis=plt.subplots(figsize=(8,11));axis.imshow(matrix,vmin=0,vmax=100,cmap='RdYlGn',aspect='auto')
    for i in range(26):
        for j in range(matrix.shape[1]):axis.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=9)
    axis.set_xticks(range(matrix.shape[1]),[f'{s}\n{state}' for s in splits for state in ['lama','baru']]);axis.set_yticks(range(26),labels)
    axis.set_title('Persentase fitur valid per huruf (%)\nGambar dan label tetap sama; test tidak dibuka')
    fig.tight_layout();fig.savefig(output/'per_letter.png',dpi=150);plt.close(fig)
    lines=['# Hasil pemulihan landmark BISINDO','','Recovery otomatis tanpa kotak/label acuan saat deteksi; rotasi/ROI dipetakan kembali ke koordinat gambar asli. Hasil sudah melalui filter duplikasi rangka. Tidak ada training model ulang.','',
           '| Split | Gambar | Valid lama | Valid baru | Persentase lama → baru | Gagal lama → baru | Dua tangan lama → baru |',
           '|---|---:|---:|---:|---:|---:|---:|']
    details=[]
    for s in splits:
        r=report['splits']['bisindo/'+s];b=r['baseline'];n=r['total']
        lines.append(f"| {s} | {n:,} | {b['feature_valid']:,} | {r['feature_valid']:,} | {b['feature_valid_rate']*100:.2f}% → {r['feature_valid_rate']*100:.2f}% | {n-b['feature_valid']} → {n-r['feature_valid']} | {b['hand_counts'].get('2',0)} → {r['hand_counts'].get('2',0)} |")
        retried=r.get('quality_retried',r.get('duplicate_proposals_retried',0))
        details.extend(['',f"{s}: kandidat dua rangka yang terlalu berdekatan dicoba ulang pada {retried} gambar; transisi akhir {r['transitions']}."])
    lines.extend(details)
    if report.get('policy_sha256'):
        lines.extend(['','V2 memakai pemeriksaan jarak titik sepadan dan kumpulan titik tanpa mengandalkan nomor jari. Tidak ada pengecualian per gambar atau koreksi manual. Contoh X train #5977 dan O validation #1457 ditolak melalui aturan otomatis yang sama dengan runtime. Flag review_required tetap tersedia; tidak seluruh landmark disahkan secara visual.'])
    else:
        lines.extend(['', 'Satu hasil validation dikembalikan ke baseline melalui tinjauan teknis manual tercatat. Ini laporan pemulihan data semiotomatis, bukan pengukuran akurasi detektor otomatis. Inspeksi grid train masih menemukan kandidat X dengan rangka bertumpuk; review_required tetap aktif dan cache belum merupakan anotasi bersih terverifikasi.'])
    lines.extend(['','![Ringkasan](overview.png)','','![Per huruf](per_letter.png)','','Angka valid hanya pemeriksaan geometri. Satu tangan belum tentu pose lengkap; deteksi dua tangan belum membuktikan ketepatan semua ruas jari. Tidak ada gambar yang dibuang untuk menaikkan persentase.','',
                  f'Cache final: {source.as_posix()}. Cache proposal dan baseline tetap disimpan. Overlay before/after dan CSV per gambar ada di direktori cache final.','',
                  'Metode ini menambah biaya inferensi karena beberapa view dicoba. Perbandingan waktu fallback bukan FPS webcam; proses_ms lama berasal dari cache baseline. Sebelum training ulang, praproses offline dan webcam harus memakai kebijakan recovery/filter yang sama.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\n'.join(lines[:12]).encode('ascii', errors='replace').decode('ascii'))


if __name__=='__main__':main()
