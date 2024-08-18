% Matrix

百度更早的容器部署系统。

Matrix 概念：

- user —— user_name、token
- pool —— pool_name
- user_pool —— user_name、pool_name
- host —— host_id、content，content中包括hostId、hostName、tagList、state、poolName
- service —— service_name、content，content中包括serviceName、meta信息和userName
- instance —— service_name、offset、content，content包括serviceName、offset、HostId、meta信息和资源信息等
