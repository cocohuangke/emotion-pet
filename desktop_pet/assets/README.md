# 宠物美术素材目录（占位说明）

本目录用于存放桌面宠物的**美术素材帧图**。宠物外观由 `desktop_pet/emotion_display.py` 在
运行时**程序化生成占位贴图**（渐变圆脸 + 文字表情 + 精力条/等级徽章）。

## 预期目录结构

```
assets/
├── happy/
│   ├── frame_00.png      # 情绪帧序列，命名 frame_00.png ~ frame_nn.png
│   ├── frame_01.png
│   ├── frame_02.png
│   └── frame_03.png
├── sad/
│   └── ...               # 每类情绪一个子目录，共 7 类
├── angry/
├── fear/
├── surprise/
├── disgust/
└── neutral/
```

## 约定

- 每个情绪子目录对应 `config.yaml` 中 `emotion_labels` 的一类；
- 帧命名采用 `frame_{i:02d}.png`（从 `00` 开始），建议 4 帧循环；
- 帧尺寸建议 220×220（与 `PetWindow.DEFAULT_SIZE` 一致）；
- `PetAnimation.load_asset_frames(emotion, asset_root)` 会自动加载上述
  命名规则的帧图；若目录缺失或加载为空，则回退到程序化占位贴图。

## 占位图来源

在未放置素材时，请勿手动创建空图片；运行时渲染的占位贴图由
`EmotionDisplay.render_frames()` 生成，足以支撑交互演示。
