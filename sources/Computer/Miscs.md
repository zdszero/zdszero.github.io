% 记录一些恼人的配置和工具记录

### chromium 在 i3wm 中无法加载密码

- chrome://flags
- 搜索 Removes passwords that can no longer be decrypted，设置为 Enabled
- 如果可以的话，设置登录时的 password manager，选择 kwallet 或者 gnome-keyring
- 设置 keyring managing daemon 为开机自动启动
