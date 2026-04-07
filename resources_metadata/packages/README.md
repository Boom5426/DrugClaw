# Resource Packages

`resources_metadata/packages/` 保存 resource package manifest。

- 每个 manifest 使用 JSON。
- 路径字段优先写仓库相对路径，例如 `resources_metadata/...`。
- manifest 只描述组件位置与依赖，不保存大文件本体。
- 当前最小字段：
  - `package_id`
  - `skill_name`
  - `dataset_bundle`
  - `protocol_docs`
  - `how_to_docs`
  - `software_dependencies`
  - `knowhow_docs`
