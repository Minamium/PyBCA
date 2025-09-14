---
title: BCL Guide (v0.1)
nav_order: 10
---

# Brownian Circuits Language(BCL)

ブラウン回路アーキテクチャをテキストベースで記述するためのフレームワーク.

素子より任意の回路モジュールを構成, そのI/Oをつなげて回路を構築する. 回路モジュール内でのみ絶対座標を意識して, モジュール同士の接続と回路構築は相対座標記述でできるように.

## Global input / output の定義

input / output の定義. 

定義した回路モジュールのIOをつなげる.

```bcl

input: x_in, y_in
output: z_out

```

## 回路モジュール宣言

定義する回路モジュールのname, ID, sizeを宣言する.

```bcl

construct info(name: "Module_A", ID: 0, Glid_size: 10*10){
    module_input: a, b
    module_output: c
}

```


