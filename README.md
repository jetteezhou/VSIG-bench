# Visual-Speech Intent Grounding (VSIG) Benchmark

这是一个针对 **Visual-Speech Intent Grounding (VSIG)** 任务的推理与评估基准项目。

## 任务介绍
VSIG 要求智能体作为积极的倾听者和观察者，根据第一人称视频和用户语音指令：
1. **意图 Grounding**: 将模糊的潜在意图解码为明确的可执行指令。
2. **空间 Grounding**: 输出目标物体 (Target Object) 和空间示能点 (Spatial Affordance) 的精确坐标。
3. **时间 Grounding**: 输出用户意图描述的目标物体和空间示能点的时间戳

## 项目结构
```
.
├── data/                           
│   └── 四楼测试区_李凯越_李尚儒_拍摄/  
│       └── 指令1/ 
│       └── 指令2/ 
│       └── 指令3/ 
│       └── 指令4/ 
│       └── 指令5/ 
│       └── 指令6/ 
│   └── 四楼实训区_场景1_康昱灵_张潇杨_拍摄/  
│       └── 指令1/ 
│       └── 指令2/ 
│       └── 指令3/ 
│       └── 指令4/ 
│       └── 指令5/ 
│       └── 指令6/          
├── src/
│   ├── prompts/            # Prompt 设计
│   ├── models/             # 模型接口 (VLM Wrapper)
│   ├── utils/              # 视频处理工具
│   └── eval/               # 评估指标代码
├── config.py               # 参数配置文件
├── main.py                 # 推理入口脚本
└── requirements.txt        # 依赖列表
```


不同的指令数据文件夹下用户的指向性动作和说话内容分别问：

| 指令 | 指向性动作描述 | 说话内容 | 示例 | 意图grounding | 空间grounding | 时间grounding |
|------|---------------|----------|------|------|------|------|
| 指令1 | 用户的手指向一个物体 | 不说话 | 用户没说话、手指向一个苹果然后放下来了 | 一个苹果  | vlm预测最后一帧图像中苹果的point是否在标注的苹果的mask内 | - |
| 指令2 | 用户的手指向一个物体 | "把这个拿起来" | 用户说'把这个拿起来'的同时指向一个评估，然后放下手 | 把苹果拿起来 | vlm预测最后一帧图像中苹果的point是否在标注的苹果的mask内 | vlm预测'苹果'的时间戳是否在用户说'这个'的时间范围内 |
| 指令3 | 用户的手指向一个物体然后又指向一个可放置的区域 | "把这个放到这里" | 用户一边说'把这个'一边指向一个苹果、然后连贯的一边说'放在这里'一边指向桌子上的一个空闲区域 | 把苹果放在桌子上（纸巾盒右边的）空间区域 | vlm预测最后一帧图像中苹果的point是否在标注的苹果的mask内，以及预测的放置的point和标注的空闲区域的points的距离 | vlm预测'苹果'的时间戳是否在用户说'把这个'的时间范围内，以及预测放置空间的时间戳是否在用户说'放在这里'的时间范围内 |
| 指令4 | 用户的手指向一个物体然后又指向一个物体 | "把这个放到它的右边" | 用户一边说'把这个'一边指向一个苹果、然后连贯的一边说'放在它的右边'一边指向一个香蕉 | 把苹果放在香蕉的右边 | vlm预测最后一帧图像中苹果的point是否在标注的苹果的mask内，以及预测的放置区域的point和标注的香蕉右侧的区域的points的距离 | vlm预测'苹果'的时间戳是否在用户说'把这个'的时间范围内，以及预测放置空间的时间戳是否在用户说'放在它的右边'的时间范围内 |
| 指令5 | 用户的手指向一个物体然后又指向一个物体、随后再指向另一个物体 | "把这个放到它的右边，然后把它拿起来" | 用户一边说'把这个'一边指向一个苹果、然后连贯的一边说'放在它的右边'一边指向一个香蕉、然后再连贯的一边说'然后把它拿起来'一边指向一个杯子 | 把苹果放在香蕉的右边，然后把杯子拿起来 | vlm预测最后一帧图像中苹果的point是否在标注的苹果的mask内，以及预测的苹果放置区域的point和标注的香蕉右侧的区域的points的距离，以及预测的杯子的point是否在标注的杯子的mask内 | vlm预测'苹果'的时间戳是否在用户说'把这个'的时间范围内，以及预测苹果的放置空间的时间戳是否在用户说'放在它的右边'的时间范围内，以及预测的'杯子'的时间戳是否在用户说'然后把它拿起来'的时间范围内 |
| 指令6 | 用户的手指向一个物体然后又指向一个物体、随后再指向另一个物体、然后再指向第四个物体 | "把这个放到它的右边，然后把它放在它的左边" | 用户一边说'把这个'一边指向一个苹果、然后连贯的一边说'放在它的右边'一边指向一个香蕉、然后再连贯的一边说'然后把它'一边指向一个杯子、再一边说'放在它的左边'一边指向一个碗 | 把苹果放在香蕉的右边，然后把杯子放到碗的左边 | vlm预测最后一帧图像中苹果的point是否在标注的苹果的mask内，以及预测的苹果放置区域的point和标注的香蕉右侧的区域的points的距离，以及预测的杯子的point是否在标注的杯子的mask内，以及预测的杯子的放置区域和标注的碗的左侧区域的points之间的距离 | vlm预测'苹果'的时间戳是否在用户说'把这个'的时间范围内，以及预测苹果的放置空间的时间戳是否在用户说'放在它的右边'的时间范围内，以及预测的'杯子'的时间戳是否在用户说'然后把它'的时间范围内，以及预测的杯子的放置区域的时间戳是否在用户说'放在它的左边'的时间范围内 |

注意每个数据中用户说的内容都有一些细微的差别，asr_result会提取出每个词语的时间戳，如果asr_result为null则不统计时间grouding的得分。

每个指令的内容会有区别，但是标注的顺序是完全一致。
指令1的标注示例：
```json
{
    "asr_result": null,
    "folder": "C:\\Users\\lenovo\\Desktop\\四楼实训区_场景1_康昱灵_张潇杨_拍摄\\指令1",
    "id": "1766825947329",
    "is_invalid": false,
    "object_space": [
      {
        "name": "白色蜂窝状的长方体",
        "points": [
          [
            906,
            394
          ],
          [
            880,
            405
          ],
          [
            905,
            421
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBORw0KGgoAAAAN...",
          "bbox": [
            707,
            932,
            855,
            1044
          ],
          "score": 0.6937909722328186,
          "point_on_mask": true
        }
      }
    ],
    "scene": "四楼实训区",
    "task_template": "指令1",
    "video_name": "VID20251227151338.mp4"
}
```

指令2的标注示例：
```json
{
    "asr_result": null,
    "folder": "C:\\Users\\lenovo\\Desktop\\四楼实训区_场景1_康昱灵_张潇杨_拍摄\\指令2",
    "id": "1766826259100",
    "is_invalid": false,
    "object_space": [
      {
        "name": "透明外壳的盒子",
        "points": [
          [
            770,
            376
          ],
          [
            751,
            410
          ],
          [
            779,
            404
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBORw0K...",
          "bbox": [
            662,
            766,
            879,
            911
          ],
          "score": 0.7166661024093628,
          "point_on_mask": true
        }
      }
    ],
    "scene": "四楼实训区",
    "task_template": "指令2",
    "video_name": "VID20251227151818.mp4"
  }
```

指令3的标注示例：
```json
{
    "asr_result": null,
    "folder": "C:\\Users\\lenovo\\Desktop\\四楼实训区_场景1_康昱灵_张潇杨_拍摄\\指令3",
    "id": "1766826547511",
    "is_invalid": false,
    "object_space": [
      {
        "name": "绿色圆形柱体",
        "points": [
          [
            848,
            400
          ],
          [
            813,
            414
          ],
          [
            863,
            418
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBORw0...",
          "bbox": [
            754,
            862,
            830,
            983
          ],
          "score": 0.8961913585662842,
          "point_on_mask": true
        }
      },
      {
        "name": "绿色圆形柱体和橙色外壳的万用表的中间空闲区域",
        "points": [
          [
            792,
            319
          ],
          [
            779,
            344
          ],
          [
            819,
            346
          ]
        ],
        "type": "space"
      }
    ],
    "scene": "四楼实训区",
    "task_template": "指令3",
    "video_name": "VID20251227153315.mp4"
  }
```

指令4的标注示例：
注意指令4和benchmark规定的标注不完全一致，标注的object_space的顺序是[object1, object2, space1]，这是为了方便可视化，实际上指令4的gt是一个物体和一个放置区域，也就是评估的时候跳过object2，并且把object2的name和space1合并一下。例如下面的示例是[白色蜂窝状的长方体, 银色圆形中间镂空柱体, "左面"]，那么评估意图grounding的时候让模型生成一个目标物体和一个目标区域然后分别和白色蜂窝状的长方体与银色圆形中间镂空柱体的左面对比（注意这里直接把object2的name和space1的name合并用于意图grounding的评估），评估空间grounding的时候只让模型生成一个目标物体和一个目标区域，然后分别和白色蜂窝状的长方体的mask与"左面"的points对比（注意这里就直接跳过object2的mask）。

```json
{
    "asr_result": null,
    "folder": "C:\\Users\\lenovo\\Desktop\\四楼实训区_场景1_康昱灵_张潇杨_拍摄\\指令4",
    "id": "1766827449269",
    "is_invalid": false,
    "object_space": [
      {
        "name": "白色蜂窝状的长方体",
        "points": [
          [
            703,
            530
          ],
          [
            684,
            557
          ],
          [
            712,
            554
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBORw0K...",
          "bbox": [
            1005,
            722,
            1112,
            811
          ],
          "score": 0.8547917604446411,
          "point_on_mask": true
        }
      },
      {
        "name": "银色圆形中间镂空柱体",
        "points": [
          [
            621,
            449
          ],
          [
            589,
            455
          ],
          [
            621,
            466
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBOR...",
          "bbox": [
            848,
            624,
            914,
            710
          ],
          "score": 0.8755226135253906,
          "point_on_mask": true
        }
      },
      {
        "name": "左面",
        "points": [
          [
            706,
            389
          ],
          [
            680,
            404
          ],
          [
            712,
            414
          ]
        ],
        "type": "space"
      }
    ],
    "scene": "四楼实训区",
    "task_template": "指令4",
    "video_name": "VID20251227154155.mp4"
  }

```

指令5的标注示例：
评估方法和指令4类似，跳过object2，讲obejct2和name和space1的name合并

```json
{
    "id": "1766830113620",
    "folder": "C:\\Users\\lenovo\\Desktop\\四楼实训区_场景1_康昱灵_张潇杨_拍摄\\指令5",
    "video_name": "VID20251227154646.mp4",
    "task_template": "指令5",
    "scene": "四楼实训区",
    "object_space": [
      {
        "name": "白色长条形的打印件",
        "points": [
          [
            666,
            506
          ],
          [
            672,
            525
          ],
          [
            690,
            542
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBORw...",
          "bbox": [
            952,
            700,
            1069,
            787
          ],
          "score": 0.8658263087272644,
          "point_on_mask": true
        }
      },
      {
        "name": "银色圆形中间镂空柱体",
        "points": [
          [
            640,
            417
          ],
          [
            680,
            411
          ],
          [
            678,
            425
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBOR...",
          "bbox": [
            770,
            678,
            840,
            767
          ],
          "score": 0.8700471520423889,
          "point_on_mask": true
        }
      },
      {
        "name": "右面",
        "points": [
          [
            601,
            448
          ],
          [
            612,
            474
          ],
          [
            630,
            457
          ]
        ],
        "type": "space"
      },
      {
        "name": "白色圆形中间镂空柱体",
        "points": [
          [
            826,
            452
          ],
          [
            860,
            446
          ],
          [
            855,
            462
          ]
        ],
        "type": "object",
        "mask": {
          "mask_base64": "iVBOR...",
          "bbox": [
            834,
            886,
            907,
            960
          ],
          "score": 0.85821133852005,
          "point_on_mask": true
        }
      }
    ],
    "is_invalid": false,
    "asr_result": null
  }

```


指令6的标注示例为：
和指令4与5的评估方法类似，但要注意跳过obejct2白色圆形中间镂空柱体的mask，合并obejct2和space1的name为'白色圆形中间镂空柱体的前面'，并且要跳过object4黑色长方体打印件的mask，合并obejct4和space2的name为'黑色长方体打印件的后面'

```json
{
    "asr_result": {
      "sentences": [
        {
          "begin_time": 3710,
          "end_time": 10110,
          "text": "把这个放到它的前面，然后再把这个放到它的后面。"
        }
      ],
      "text": "把这个放到它的前面，然后再把这个放到它的后面。",
      "words": [
        {
          "begin_time": 3710,
          "end_time": 4670,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "把这个"
        },
        {
          "begin_time": 5230,
          "end_time": 5590,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "放到"
        },
        {
          "begin_time": 5590,
          "end_time": 5870,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "它的"
        },
        {
          "begin_time": 5870,
          "end_time": 6630,
          "fixed": false,
          "punctuation": "，",
          "speaker_id": null,
          "text": "前面"
        },
        {
          "begin_time": 7310,
          "end_time": 7790,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "然后再"
        },
        {
          "begin_time": 7790,
          "end_time": 8710,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "把这个"
        },
        {
          "begin_time": 9230,
          "end_time": 9630,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "放到"
        },
        {
          "begin_time": 9630,
          "end_time": 9910,
          "fixed": false,
          "punctuation": "",
          "speaker_id": null,
          "text": "它的"
        },
        {
          "begin_time": 9910,
          "end_time": 10110,
          "fixed": false,
          "punctuation": "。",
          "speaker_id": null,
          "text": "后面"
        }
      ]
    },
    "folder": "C:\\Users\\lenovo\\Desktop\\四楼实训区_场景1_康昱灵_张潇杨_拍摄\\指令6",
    "id": "1766831791974",
    "is_invalid": false,
    "object_space": [
      {
        "mask": {
          "bbox": [
            1031,
            660,
            1118,
            715
          ],
          "mask_base64": "iVBOR...",
          "point_on_mask": true,
          "score": 0.8168432712554932
        },
        "name": "红色手柄的螺丝刀",
        "points": [
          [
            618,
            577
          ],
          [
            636,
            555
          ]
        ],
        "type": "object"
      },
      {
        "mask": {
          "bbox": [
            633,
            769,
            718,
            846
          ],
          "mask_base64": "iVBO...",
          "point_on_mask": true,
          "score": 0.8550975322723389
        },
        "name": "白色圆形中间镂空柱体",
        "points": [
          [
            725,
            348
          ],
          [
            757,
            350
          ]
        ],
        "type": "object"
      },
      {
        "name": "前面",
        "points": [
          [
            809,
            420
          ],
          [
            835,
            403
          ],
          [
            837,
            425
          ]
        ],
        "type": "space"
      },
      {
        "mask": {
          "bbox": [
            786,
            593,
            850,
            666
          ],
          "mask_base64": "iVBORw0K...",
          "point_on_mask": true,
          "score": 0.7713631987571716
        },
        "name": "蓝色圆形柱体",
        "points": [
          [
            561,
            425
          ],
          [
            591,
            427
          ]
        ],
        "type": "object"
      },
      {
        "mask": {
          "bbox": [
            920,
            457,
            1012,
            543
          ],
          "mask_base64": "iVBORw0...",
          "point_on_mask": true,
          "score": 0.9305926561355591
        },
        "name": "黑色长方体打印件",
        "points": [
          [
            438,
            503
          ],
          [
            462,
            486
          ],
          [
            468,
            511
          ]
        ],
        "type": "object"
      },
      {
        "name": "后面",
        "points": [
          [
            415,
            431
          ],
          [
            413,
            451
          ],
          [
            441,
            441
          ]
        ],
        "type": "space"
      }
    ],
    "scene": "四楼实训区",
    "task_template": "指令6",
    "video_name": "VID20251227155159.mp4"
  }

```



## 安装
```bash
pip install -r requirements.txt
```

## 配置
项目的所有配置（模型参数、路径设置等）均在 `config.py` 中管理。
请打开 `config.py` 并根据需要修改：

```python
class Config:
    # --- 模型配置 ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") 
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = "gpt-4o"
    # ...
```

## 运行
配置好 `config.py` 后，直接运行：

```bash
python main.py
```

如果未提供 API Key，程序会报错退出。建议通过环境变量设置 Key，或者临时在 `config.py` 中修改。

## 部分结果

### Qwen3-Omni-30B-A3B-Instruct
总样本数为180个，处理成功的样本数为174个，有6个失败是因为模型输出的json格式解析有问题，该模型在benchmark指标下的具体结果如下：

    "intent_grounding_accuracy": 0.07436781609195403,
    "spatial_grounding_accuracy": 0.015957446808510637,
    "temporal_grounding_accuracy": 0.0,
    "overall_score": 0.04516263145023233
结果上看，Qwen3-Omni-30B-A3B-Instruct对这类第一人称视频+手势指向的任务理解能力不强。