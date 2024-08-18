% 接入层

### 整体架构

```
USER ---- DNS/CDN ---- TDO
 |
 |
BGW
 |
 |
BFE
```

#### DNS/CDN/TDO

- DNS：Domain Name System，域名到 IP 的映射服务，找到服务地址
- CDN：Content Delivery Network，内容分发网络，依靠部署在各地的边缘服务器，通过中心平台的负载均衡、内容分发、调度等功能，使用户就近选择服务器获取内容，降低网络拥塞，提高用户访问速度和命中率。快速响应
- TDO：Traffic Distribution Optimizer，流量分配优化，流量的负载均衡，通过 traceroute 发掘对用户而言，速度最优的 IDC，每周更新 CDNDNS 域名和 IP 地址映射关系
    - 读取 CDNDNS 日志，按照 C 类地址对日志中用户的 IP 地址进行聚合
    - 每天 3 次对聚类的结果使用 traceroute 探测到达对应 IP 地址类的时间，并把当天 3 次结果求平均值
    - 根据每天每个 IDC 探测出到达 IP 地址的时间，每周更新 CDNDNS 的域名和 IP 地址映射
    - TDO 和 GTC，最终的生效，都是需要将域名解析配置，下发到 DNS 服务器

shifen.baidu.com: 域名的CNAME，多个域名的中继代理，方便域名管理、切流

#### BGW

BGW：Baidu GateWay，百度智能网关，四层负载均衡平台，分析IP层和传输层，提供统一的VIP。

VIP 中的 virtual 一次是相对于 client 来说，隐藏了复杂的请求处理响应过程，在外界看来就像是访问了单个服务器。

- 4 层的负载均衡
    - NAT模式：在转发和返回数据包时，修改其中的源IP和目的IP。
    - DR模式
    - TUN模式
- RS 保活机制
- 攻击防御

#### BFE

Baidu Frontend，百度统一前端，做七层的负载均衡，处理HTTP，流量的接入和转发，全局流量调度，安全和防攻击，数据分析

根据HTTP请求首部，确定应该将请求的集群、子集群和具体实例。

[集群、子集群和实例](/Computer/baidu_intra/concepts.md#集群)

```
upstream myapp1 {
    server backend1.example.com weight=5;
    server backend2.example.com;
    server backend3.example.com down;  # 标记为不可用
    server backend4.example.com backup;  # 备用服务器
}
```

### 整体流程

1. 用户查询域名 ip
2. 本次缓存命中返回，未命中进行 DNS 查询
3. CDN 根据实施和用户信息返回响应最快的 vip
4. 用户向 vip 发起 TCP 请求
5. 通过 BGW 攻击检测后，根据负载策略选择 RS 下发请求
6. BFE 解析 http 报文，通过攻击检测后，调度后端 PR 下发请求
7. 接收结果逐层返回用户

### 注意点

#### 负载均衡

- 硬件方法：NetScaler这种，一般是非IT企业
- DNS 轮询
- 负载均衡器
    - 四层LB
        - [LVS](https://www.alibabacloud.com/blog/load-balancing---linux-virtual-server-lvs-and-its-forwarding-modes_595724)（Linux Virtual Server）
    - 七层LB
        - nginx
