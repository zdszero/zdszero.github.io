% K8S

### Intro

abstracts away the hardware infrastructure and exposes your whole data- center as a single enormous computational resource

### Container

- linux namespace: each process sees its own personal view of the system (files, processes, network interfaces, hostname, and so on)
- cgroups (linux control groups): limit the amount of resources the process can consume (CPU, memory, network bandwidth, and so on)

__VM and docker container__

### Image

filesystem, metadata

image is composed of layers

### POD

- __Why Pod?__

resource isolation

- __Container and pod relationship? Why Pod?__

containers in the same pod shares the port and ip space.

- __Pod allocation?__

- __Why assign label to pod?__

- __Replica Set and Replication Controller__

### Service
