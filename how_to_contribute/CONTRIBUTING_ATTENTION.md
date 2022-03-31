# How To Contribute MindSpore ModelZoo

<!-- TOC -->

- [How To Contribute MindSpore ModelZoo](#how-to-contribute-mindspore-modelzoo)
    - [Preparation](#preparation)
        - [Understand Contribution Agreements And Procedures](#understand-contribution-agreements-and-procedures)
    - [Submit Code](#submit-code)
        - [CodeStyle](#codestyle)
        - [Directory Structure](#directory-structure)
        - [ReadMe](#readme)
        - [Third Party Reference](#third-party-reference)
            - [Reference Additional Python Libraries](#reference-additional-python-libraries)
            - [Reference Third-Party Open Source Code](#reference-third-party-open-source-code)
            - [Reference Other System Libraries](#reference-other-system-libraries)
        - [Submit The Self-Check List](#submit-the-self-check-list)
    - [Maintenance And Communication](#maintenance-and-communication)
        - [Signature](#signature)

<!-- TOC -->

This guidance is used to clarify the ModelZoo contribution specification to ensure that many developers can participate in the construction of ModelZoo in a relatively uniform style and process.

## Preparation

### Understand Contribution Agreements And Procedures

You should first consult MindSpore's [CONTRIBUTING.md](../CONTRIBUTING.md) instructions to understand the Open source agreement and how MindSpore works, and make sure you have signed the CLA.

<!--
### Define Goals For Your Contributions

If you want to contribute, we recommend that you start with some of the easier issues. You can find some simple bugfix jobs in the following list.

- [wanted bugfix](https://gitee.com/mindspore/mindspore/issues?assignee_id=&author_id=&branch=&issue_search=&label_ids=58021213&label_text=kind/bug&milestone_id=&program_id=&scope=&sort=newest&state=open)

If you can make independent network contributions, you can find our list of networks to implement in the list below.

- [wanted implement](https://gitee.com/mindspore/mindspore/issues?assignee_id=&author_id=&branch=&issue_search=&label_ids=58022151&label_text=device%2Fascend&milestone_id=&program_id=&scope=&sort=newest&state=open)

Remember to send a reply after the issue is selected to let others know that you are working on the issue. When you're done with something, also go back to Issue to update your work. If you have problems with the process, feel free to update your progress in the issue.
-->

## Submit Code

### CodeStyle

Reference [CONTRIBUTING.md](../CONTRIBUTING.md), you should make sure your code is consistent with MindSpore's existing CodeStyle.

For some details of implementations, you could refer to some recommended models such as [Resnet](https://gitee.com/mindspore/models/tree/master/official/cv/resnet), [Yolov5](https://gitee.com/mindspore/models/tree/master/official/cv/yolov5), [IPT](https://gitee.com/mindspore/models/tree/master/research/cv/IPT), [Transformer](https://gitee.com/mindspore/models/tree/master/official/nlp/transformer), etc.

### Directory Structure

```shell
model_zoo
├── official                                    # Officially supported models
│   └── XXX                                     # Model name
│       ├── README.md                           # Model specification document
│       ├── requirements.txt                    # Dependency documentation
│       ├── eval.py                             # Accuracy verification script
│       ├── export.py                           # Inference model export script
│       ├── scripts                             # script file
│       │   ├── run_distributed_train.sh        # Distributed training script
│       │   ├── run_eval.sh                     # Verify the script
│       │   └── run_standalone_train.sh         # Single machine training script
│       ├── src                                 # Model definition source directory
│       │   ├── XXXNet.py                       # Definition of model structure
│       │   ├── callback.py                     # Callback function definition
│       │   ├── config.py                       # Model configuration parameter file
│       │   └── dataset.py                      # Data set processing definition
│       ├── ascend_infer                        # (Optional) Scripts for offline reasoning on Ascend reasoning devices
│       ├── third_party                         # (Optional) Third-party code
│       │   └── XXXrepo                         # (Optional) Complete code cloned from a third-party repository
│       └── train.py                            # Training script
├── research                                    # Unofficial research script
├── community                                   # Partner script links
└── utils                                       # General tool for modeling
```

You can follow these guidelines and make changes to the template to suit your own implementation

1. Only executable scripts with the 'main method' are placed in the root directory of the model. The definition files of the model are placed in the 'src' directory, which can organize the hierarchy according to the complexity of the model.

2. The configuration parameters should be separated from the network definition, and all configurable parameters should be defined in the 'src/config.py' file.

3. Upload content should contain only scripts, code, and documentation, and **do not upload** any data sets, as well as directories and files generated during the run.

4. Third_party is used to store third-party code that needs to be referenced, but you should not copy the code directly to the directory and upload it. Instead, you should use the form of git link and download it when you use it.

5. The code for each model should be its own closure that can be migrated and used independently and should not rely on code outside the model directory. Utils is a generic tool, not a generic library.

6. **Do not include** any of your personal information, such as your host IP, personal password, local directory, etc.

### ReadMe

Each AI model needs a corresponding 'readme.md' documentation that describes the current model implementation and communicates the following information to other users:

1. What model is this? What are the sources and references?
2. What does the current implementation contain?
3. How to use existing implementations?
4. How does the model perform?

For this, we provide a basic [README TEMPLATE](./README_TEMPLATE.md) that you should refer to to refine your documentation, as well as README for other existing models.

### Third Party Reference

#### Reference Additional Python Libraries

Be sure to specify any additional Python libraries you need and corresponding versions (if explicitly required) in the 'requirements.txt' file. You should prioritize third-party libraries that are compatible with the MindSpore framework.

#### Reference Third-Party Open Source Code

You should make sure that the code you submit is your own original development.

When you need to leverage the power of the open source community, you should first use mature and trusted open source projects and verify that the open source license of your chosen open source project meets the requirements.

When you use open source code, the correct way to use it is to get the code from your Git address and archive it in a separate 'third_party' directory to keep it isolated from your own code. **Do not copy the corresponding code snippets into your own submission.**

#### Reference Other System Libraries

You should reduce your reliance on unique system libraries, as this often means that your commit is hard to reuse across different systems.

When you do need to use some unique system dependencies to get things done, you need to specify how to get and install them in the instructions.

### Submit The Self-Check List

Your submitted code should be fully reviewed and self-checked by referring to the following checklist

- [ ] Code style conforms to specification
- [ ] Code adds comments where necessary
- [ ] The document has been synchronized
- [ ] Synchronously adds the necessary test cases
- [ ] All third party dependencies are explained, including code references, Python libraries, data sets, pre-trained models, etc
- [ ] The project organization complies with the requirements in [Directory Structure](#directory-structure).
- [ ] Complete readme writing and pass CI tests

## Maintenance And Communication

We appreciate your contribution to the MindSpore community, and we encourage you to keep an eye on your code after you complete a submission. You can mark your signature, email address and other contact information in the README of the submitted model, and keep an eye on your Gitee and Github information.

Other developers may be using the model you submitted and may have some questions during use. In this case, you can communicate with you in detail through issues, in-site messages, emails, etc.

### Signature

You could sign your name, your homepage on gitee or github, and your organization in readme just like the chapter *Contributor* in the readme template.
