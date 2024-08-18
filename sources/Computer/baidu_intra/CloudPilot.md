---
title: CloudPilot
theme: latex
---

## 查询

### 元数据

API: `http://10.146.211.21:8031/res/capacity/getappmeta`

```
"app_meta": {
    "pass_type": PASS平台类型,
    "product": 产品线,
    "module": 模块名,
    "app_id": 服务ID,
    "cluster": 集群,
    "account_id": 资源账号,
    "service_level": 服务等级,
    "bns": BNS,
    "disk_type": 磁盘类型,
    "disk_exclusive": 磁盘是否独占,
    "limit_cpu_usage": CPU使用限制,
    "extend_join": 是否接入自动扩容,
    "shrink_join": 是否接入自动缩容,
    "extend_mask": 是否豁免自动扩容,
    "shrink_mask": 是否豁免自动缩容,
    "rescale_direction_prior": 伸缩的优先方向,
    "owner": 负责人,
    "manager": 负责人经理
}
```

### 可用度

API: `http://10.146.211.21:8031/pandora/getappminusable`

```
"min_usable_info": {
    @TODO
    "instanceControlBudget": {
        "type": "MinAvailableRatio"
        "value": 80
    }
    "minUsable": 最小可用度,
    "replica": 副本数量,
    "replica_that_at_least_need_to_keep": 至少需要维持的实例数量
}
```

### 各状态实例

API: `http://10.146.211.21:8031/feiyun/get/instance`

```
{
    "app_id": 服务ID,
    "total_num": "实例总数",
    "status_instance_top_five": {
        // RUNNING 正常运行，!RUNNING 表示非正常运行，ALL获取全部实例
        "NEW": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "DEPLOYING": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "DEPLOY_OK": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "RESTARTING": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "RUNNING": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "STOPPING": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "STOP": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "UNKNOWN": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "DESTORY": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        },
        "INVALID_STATE": {
            "value": 总数,
            "top_five_list": 排名前五的实例名称
        }
    }
}
```

### 硬限

```
{
    "cpu": 是否开启CPU硬限,
    "cpu_quota":
    "cpu_limit",
    "cpu_....": CPU硬限倍数
    "memory":
    "disk"
}
```

### 被屏蔽实例

```
{
    "disabled_instance_id_list": 被屏蔽的实例列表
}
```

### 实例状态信息

API: `http://10.146.211.21:8031/paas/instance/getinfo`

```
{
    "instance_info_list": [
        {
            "instance_id":  实例ID,
            "host_ip": pod的IP地址,
            "main_port": 端口,
            "proc_state": 进程状态,
            "container_state": 容器状态,
            "pass_agent_state": Pass Agent 状态,
            "frozen": 是否被冻结,
            "bsn_disabled": BNS被禁止,
            "under_repair": 正在修复,
            "migrating": 是否在迁移
        }
    ]
}
```

### 通过IP和端口查询实例名

API: `http://10.146.211.21:8031/shell/get/instance`

### 服务监控数据

API: `http://10.146.211.21:8031/alm/selfhealing/geteventlist`

## Instance 操作

### 重启

API: `http://10.146.211.21:8031/feiyun/action/instance/restart`

```
{
    "success_instances": 成功实例列表,
    "failed_instances": 失败实例列表
}
```

### 屏蔽

API: `http://10.146.211.21:8031/feiyun/action/instance/disable`

### 解屏蔽

### 迁移

API: `http://10.146.211.21:8031/feiyun/action/instance/migrate`

### 删除

API: `http://10.146.211.21:8031/feiyun/action/instance/delete`

### 维修

### 停止

### 启动

### 打断

### 解冻
