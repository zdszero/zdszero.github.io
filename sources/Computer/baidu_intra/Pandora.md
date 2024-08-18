% Pandora

云计算三层：SaaS、PaaS、IaaS。
IaaS 提供基础的计算资源，包括虚拟机、存储、网络和操作系统。
PaaS 提供一个平台，允许开发人员在其上构建、部署和管理应用程序，而无需处理底层的基础设施
SaaS 提供完整的软件解决方案，用户通过互联网访问和使用软件，而无需管理底层的硬件、操作系统或应用程序软件。

不负责创建容器，Matrix负责创建容器，Pandora负责调度业务的实例，告诉Matrix去某一台机器创建。

服务全生命周期管理。

| Name         | Meaning                                                     |
|--------------|-------------------------------------------------------------|
| app          | homogeneous instances' template in cluster                  |
| app tag      |                                                             |
| label        |                                                             |
| instance     | minimal service unit, corresponding to a group of processes |
| container    | isolated environment                                        |
| availability |                                                             |
| 数据项       |                                                             |
| workpath     | '='                                                         |
