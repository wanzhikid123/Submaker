# H.265 画质、体积与速度实测

2026-10-19，本机双 Tesla V100，FFmpeg 7.0.2。没有修改用户保存的编码设置、生产任务或原视频；没有调用 LLM 或上传影片。

在已测组合中，建议保留 **H.265 + CQ24 + 紧凑体积**。批量处理可用 **slower · P6**，追求最后一点指标可用 **very slow · P7**；P6/P7 的体积基本相同。希望进一步缩小时，可试 **CQ25**，代价是一定画质损失。CQ26/28 更小，但不适合承诺“与原片相同画质”。

## 软件里怎么设

“API 与高级设置 → 视频编码”：

| 设置 | 建议 |
|---|---|
| 输出编码 | H.265 / HEVC |
| GPU 速度预设 | 批量用 slower · P6；保留最高指标用 very slow · P7 |
| 画面质量 | 24 CQ；更重视体积时试 25 CQ |
| 体积策略 | 紧凑体积 |
| 编码与解码 | 保持 GPU |
| 目标视频码率 | 紧凑模式下不使用此输入框；仅“指定视频码率”模式生效 |

CQ 越小通常越清晰、文件越大；P7 比 P6 慢，不代表一定更小。NVENC 的 CQ 不能直接当作 x265 的 CRF，也不能用相同 CQ 数字认定 H.264/H.265 画质相同。[NVIDIA 编码指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html)

## 六段 20 秒复核

素材为已压缩的 H.264 High、1280×720、8-bit 4:2:0 SDR、约29.97fps，视频平均码率约1.48Mbps。从片中不同位置选取低/高运动和不同细节量的场景，使用相同中性字幕烧录后比较。各配置均测六段20秒，共36组。

| H.265 设置 | 平均 VMAF | 平均 SSIM | 抽样视频体积／原视频轨 | 最差场景平均 VMAF |
|---|---:|---:|---:|---:|
| P7 · CQ24 · 紧凑 | 95.06 | 0.988975 | 94.46% | 94.34 |
| P6 · CQ24 · 紧凑 | 94.98 | 0.988919 | 94.34% | 94.29 |
| P7 · CQ25 · 紧凑 | 94.38 | 0.988084 | 83.72% | 93.61 |
| P7 · CQ26 · 紧凑 | 93.66 | 0.987132 | 74.65% | 92.77 |
| P7 · CQ31 · 优先质量 | 95.46 | 0.989221 | 99.57% | 94.09 |
| P7 · CQ32 · 优先质量 | 94.75 | 0.988262 | 87.37% | 93.45 |

体积比例按相同窗口的总编码字节／原视频包总字节计算，包含小量测试MKV开销，不包含复制的音频。音频不重编码，因此整片缩小比例还会受到音频占比影响。CQ25 的抽样视频比当前 CQ24 再小约11.4%，但指标也降低；CQ26 再降画质更明显。

当前已完成成品采用 P7/CQ24/紧凑，实测整片大小为源文件的约90.22%，即缩小约9.78%。抽样比例与整片比例不同，不能把抽样值当作所有影片的成品大小保证。

VMAF 不是“保留了百分之多少画质”。它与 SSIM/PSNR用于比较，本次没有进行真人盲测；有损重编码不能保证与源视频逐像素相同。无损对照的 SSIM=1、PSNR=∞、VMAF约99.97，验证了帧配对及相同字幕比较流程。[VMAF FAQ](https://github.com/Netflix/vmaf/blob/master/resource/doc/faq.md)

## 为何不直接用“优先质量 + CQ24”

另一轮四处12秒初筛共52组中，P7/CQ24/优先质量平均VMAF为98.18，但视频体积为原视频轨的约239%；CQ28仍约143%，CQ30约111%。这能减少再次编码的误差，但不符合“文件更小”的目标。

软件的“紧凑体积”同时使用CQ与源视频参考码率上限。当前实现的HEVC参考目标为 `源视频码率 × 0.60 × 2^((26−CQ)/6)`，平均目标最多为源码率的90%，峰值上限最多为源码率；NVENC实际大小还受内容、CQ和缓冲约束影响。“优先质量”取消平均/峰值限制，让复杂画面使用更多码率，所以可能比原片大。

## P6 与 P7 速度

在同一个30秒片段上单卡顺序交替运行，避免画质计算与另一张卡同时工作的干扰，每种预设重复三次：

| CQ24 / 紧凑 | 三次中位耗时 | 测试视频字节 |
|---|---:|---:|
| P7 / fullres multipass | 3.909秒 | 5,077,310 |
| P6 / qres multipass | 2.577秒 | 5,072,965 |

本次P6吞吐约为P7的1.52倍，耗时少约34%，体积仅少约0.09%；六场景平均VMAF低约0.08。此处为已缓存源片段、仅视频与字幕的局部编码测速，不是整片含音轨I/O的耗时保证。

## 其他参数

- 现有空间AQ和32帧lookahead继续保留；不要把主观画质优化只按PSNR高低判断。[NVIDIA AQ 说明](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html)
- 两处12秒高级参数对照中，HEVC Temporal AQ均明确返回“不支持”，没有切换CPU。不能直接照搬新显卡的参数。
- 加权预测没有改善本次VMAF/SSIM，体积略增。
- 8-bit素材强制Main10没有在本次对照中形成稳定的画质/体积同时优势；维持源位深。10-bit/HDR素材不在本次适用范围。

## 复现

工具读取任务列表中最近加入的视频；只在没有运行或排队任务时开始，生产任务出现后不再启动下一次测试。临时编码视频只保留在RAM，少量ASS/VMAF文本随后删除；报告保存于被gitignore排除的`data/`。源片和成品不改写，不读取API密钥，不发起LLM请求。

```powershell
.\.venv\python.exe -s scripts/benchmark_quality.py --suite smoke --seconds 3 --offsets 1798 --gpus 0
.\.venv\python.exe -s scripts/benchmark_quality.py --suite sweep --seconds 12 --offsets 1798,5574,6833,8092 --gpus 0,1
.\.venv\python.exe -s scripts/benchmark_quality.py --suite confirm --seconds 20 --offsets 719,1798,3057,4316,5574,8092 --gpus 0,1
.\.venv\python.exe -s scripts/benchmark_quality.py --suite extra --seconds 12 --offsets 1798,8092 --gpus 0,1
.\.venv\python.exe -s scripts/benchmark_quality.py --suite speed --seconds 30 --offsets 1798 --gpus 0
```

上述时间位置仅用于本次足够长的素材；其他视频请设置其时长内的位置。测量采用原分辨率全部帧的VMAF v0.6.1、SSIM、PSNR；先对源/输出帧配对，再应用相同字幕参考，不把新字幕本身作为压缩损失。编码阶段保持软件实际时间戳路径；对齐操作只用于测量。[FFmpeg 画质测量文档](https://ffmpeg.org/ffmpeg-filters.html#libvmaf)
