# 昇思贡献指南

<!-- TOC -->

- [昇思贡献指南](#昇思贡献指南)
    - [贡献者许可协议](#贡献者许可协议)
    - [开始](#开始)
    - [贡献流程](#贡献流程)
        - [代码风格](#代码风格)
        - [Fork-Pull开发模式](#fork-pull开发模式)
        - [报告问题](#报告问题)
        - [提交PR](#提交pr)
    - [贡献模型](#贡献模型)

<!-- /TPC -->

## 贡献者许可协议

在你第一次提交代码给昇思社区之前，需要签署CLA，对于个体贡献者，详情请参照[CLA在线文档](https://www.mindspore.cn/icla)

## 开始

- 可以在[github](https://github.com/mindspore-ai/mindspore)或者[Gitee](https://gitee.com/mindspore/mindspore)仓库上进行Fork。
- 阅读[README_CN.md](README_CN.md)和[安装指导](https://www.mindspore.cn/install)来获取项目信息和编译指令

## 贡献流程

### 代码风格

为了昇思的易于审核、维护和开发，请遵循如下规范

- 代码规范

    在昇思社区，*Python*代码风格可以参考[Python PEP 8 Coding Style](https://pep8.org/)，*C++* 代码规范可以参考[Google C++ Coding Guidelines](http://google.github.io/styleguide/cppguide.html)。
    可以使用[CppLint](https://github.com/cpplint/cpplint)，[CppCheck](http://cppcheck.sourceforge.net)，[CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [Lizard](http://www.lizard.ws), [ShellCheck](https://github.com/koalaman/shellcheck) 和 [PyLint](https://pylint.org)，进行代码格式检查，建议将这些插件安装在你的IDE上。

- 单元测试

    *Python*单元测试风格建议采用[pytest](http://www.pytest.org/en/latest/)，*C++* 单元测试建议采用[Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)。测试用例的测试目的应该在命名上体现

- 重构

    我们鼓励开发者重构我们的代码，来消除[code smell](https://en.wikipedia.org/wiki/Code_smell)。所有的代码都必须经过代码风格检验、测试检验，重构代码也不例外。[Lizard](http://www.lizard.ws)阈值，对于nloc((lines of code without comments)是100，cnc (cyclomatic complexity number)是20，如果你收到一个*Lizard*警告，你必须在合入仓库前重构你的代码。

- 文档

    我们使用*MarkdownLint*来检查markdown文档的格式。昇思的门禁在基于默认配置的情况下修改了如下的规则：
    - MD007（无序列表缩进）：**indent**参数设置为**4**，即所有无序列表的内容都是缩进四个字节。
    - MD009（行末空格）：**br_spaces**参数设为**2**，即行末可以有0个或者两个空格。
    - MD029（有序列表序号）：**style**参数设置为**ordered**，即有序列表按照升序排列

    具体细节，请参考[RULES](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md)。

### Fork-Pull开发模式

- Fork昇思仓库

    在提交昇思项目代码之前，请确保昇思已经被fork到你自己的仓库当中，这可以使得昇思仓库和你的仓库并行开发，因此请确保两者之间的一致性。

- Clone远程仓库

    如果你想下载代码到你的本地机器，请使用git。

    ```shell
    # For GitHub
    git clone https://github.com/{insert_your_forked_repo}/mindspore.git
    git remote add upstream https://github.com/mindspore-ai/mindspore.git
    # For Gitee
    git clone https://gitee.com/{insert_your_forked_repo}/mindspore.git
    git remote add upstream https://gitee.com/mindspore/mindspore.git
    ```

- 本地代码开发

   为了保证并行分支之间的一致性，在开发代码前请创建一个新的分支：

    ```shell
    git checkout -b {new_branch_name} origin/master
    ```

   说明：origin 为昇思官方仓库，注意在创建自己仓库时尽量避免出现origin关键字，以免出现混淆

- 推送代码到远程仓库

    在更新代码之后，你需要按照如下方式来推送代码到远程仓库：

    ```shell
    git add .
    git status # Check the update status
    git commit -m "Your commit title"
    git commit -s --amend #Add the concrete description of your commit
    git push origin {new_branch_name}
    ```

- 对昇思仓库提交推送请求--提交PR

    在最后一步，你需要在你的新分支和昇思`master`分支提交一个比较请求。在这之后，Jenkins门禁会自动运行创建测试，之后你的代码会被合入到远程主仓分支上。

### 报告问题

当你遇到一个问题时,提交一个详细的问题单会对昇思有很大的贡献，我们永远欢迎填写详细、全面的issue。

当报告issue时，参考下面的格式：

- 你使用的环境(mindspore、os、python等)是什么版本的
- 这是一个BUG REPORT还是一个FEATURE REQUEST
- 这个问题的类型是什么，在issue dashbord上添加标签并进行高亮
- 发生了什么问题
- 你期望的结果是什么
- 如何复现
- 对审核人员必要的注释

**issues 警告:**

**如何确定哪一个issue是你要解决的？** 请添加一些commits在这个issue上，以此来告诉其他人你将会处理它。
**如果一个issue已经被打开一段时间了，** 建议在动手解决这个issue之前先检查一下是否还存在。
**如果你解决了一个你自己提的issue，** 在关闭之前需要让其他人知道。
**如果你希望这个issue尽快被解决，** 请给它打上标签，你可以在[label list](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md)找到不同的标签。

具体可以参照该[链接](https://gitee.com/mindspore/models/issues)下已经提交的issue示例

### 提交PR

- 在 [GitHub](https://github.com/mindspore-ai/mindspore/issues) 或者 [Gitee](https://gitee.com/mindspore/mindspore/issues) 针对一个*issue*提出你的解决方案。
- 在议题讨论和设计方案审核达成共识后，fork后完成开发后提交PR
- 提交PR后需要：签订cla，根据自检列表完成代码自检，在评论区评论/retest运行代码检查
- 贡献者的代码需要至少两个committer*LGTM*，PR才可以被允许推送，注意贡献者不允许在自己的PR上添加*LGTM*。
- 在PR被详细评审后，这个PR将会被确定能否被合入。

**PRs 警告:**

- 应避免任何无关的更改
- 确保您的提交历史记录只有一次，在确定是最终版提交记录后，将以往提交记录合并。
- 始终保持你的分支与主分支一致。
- 对于修复bug的PR，请确保链接上了issue

## 贡献模型

你可以参考[how_to_contribute](how_to_contribute)来了解如何贡献一个模型。
