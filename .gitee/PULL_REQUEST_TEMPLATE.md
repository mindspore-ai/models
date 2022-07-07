<!--  Thanks for sending a pull request!  Here are some tips for you:

1) If this is your first time, please read our contributor guidelines: https://gitee.com/mindspore/models/blob/master/CONTRIBUTING.md

2) If you want to contribute your code but don't know who will review and merge, please add label `mindspore-assistant` to the pull request, we will find and do it as soon as possible.
-->

**What type of PR is this?**
> Uncomment only one ` /kind <>` line, hit enter to put that in a new line, and remove leading whitespaces from that line:
>
> /kind bug
> /kind task
> /kind feature


**What does this PR do / why do we need it**:


**Which issue(s) this PR fixes**:
<!-- 
*Automatically closes linked issue when PR is merged.
Usage: `Fixes #<issue number>`, or `Fixes (paste link of issue)`.
-->
Fixes #

**Special notes for your reviewers**:


**CheckList**:

- [ ] I have added correct copyrights for every code file.
- [ ] I have removed all the redundant code and comments.
- [ ] I have updated or added the `requirements.txt` for the third-party libraries you need.
- [ ] I have made sure that I won't expose any personaly information such as local path with user name, local IP, etc.
- [ ] I have commented my code, particularly in hard-to-understand areas. All the comments in code files are in English.
- [ ] I have made corresponding changes to the documentation.
- [ ] I have squashed all the commits into one.
- [ ] I have test and ascertained the effect of my change in all related cases.
    - [ ] Different hardware: `CPU`, `GPU`, `Ascend910`, `Ascend310`, `Ascend310P`.
    - [ ] Different mode: `GRAPH_MODE`, `PYNATIVE_MODE`.
    - [ ] Different system: `Linux`, `Windows`, `MAC`.
    - [ ] Different number of cluster: `1pc`, `8pcs`.
- [ ] I have checked the following usual mistakes:
    - [ ] It is recommended to use `bash` instead of `sh`.
    - [ ] Don't use `type=bool` in argparse.
- [ ] If you are contributing a new model, please check:
    - [ ] I have added at least one `README`
    - [ ] The models implemented for the cloud could work offline on the local server as well.
    - [ ] I have made sure the changes for new device target will not make differences on the original one.
- [ ] If you are contributing a new function, please check:
    - [ ] I have added the corresponding information in README.
    - [ ] I have notify the maintenance team about this function.
