# ANN_Staqc

 软件工程实习作业：项目代码整改

## Changes:

* format code to match Python pep8 regulations
* fix countless typo
* project restructure: gather Models/Layers/DataProcess/others
* rename layers properly (e.g. MediumLayer -> StackingLayer)
* rename variables with reasonable names (e.g. in AttensionLayer: "a" -> enc_out; "y_trans" -> dec_out_transposed; "b" -> attension_score; etc.)
* remove unnecessary imports
* remove seeding in Layer definitions (we don't need random seeding there)
* define hyper params properly
* define paths in standalone file
* (more under-the-hood changes that I can't remember.......)
