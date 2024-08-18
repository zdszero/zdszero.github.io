% SRE

__Alternation__

- submission and approval
- annoucement
- classification
- inspection
- cut loss

__Annoucement__

- Time
- Content
- Influence
- Cause
- Handling
- Status

__Authority__

- RD: Research and Development engineer
- PM: Product Manager
- QA: Quality Assurance
- OP: Operator

__Service Layers__

- Cluster
- Domain
- Service

### Agile

### 部署

- 金丝雀部署
    - 只部署到一小部分服务器（金丝雀），这些服务器处理少量请求
    - 监控和验证
    - 逐步拓展
- 蓝绿部署
    - 维护两个环境，一个是当前环境（蓝色），另一个是备用环境（绿色）
    - 部署新版本到绿色，并进行全面测试
    - 当绿色环境测试通过后，切换流量，使所有用户访问绿色环境
    - 旧环境备用

SLA（服务级别协议）：

SLO（服务级别目标）：
