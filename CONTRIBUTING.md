We are more than happy to help you contributing to Brevitas.

Please follow the steps below and be sure that your contribution complies with our guidelines.

1. If you are looking for some issues to get started with, we have a list of <a href="https://github.com/Xilinx/brevitas/labels/good%20first%20issue">good first issues</a> in the issue tracker.

2. If you have some suggestion for features or have encoutered any bugs, don't hesitate to reach out through <a href="https://github.com/Xilinx/brevitas/issues">Brevitas Issue</a>

	We welcome submissions for:

	* New features like novel PTQ algorithms. Keep in mind that Brevitas tends to integrate new algorithms within the existing infrastructure rather than having standalone implementations
	* Support for new quantized layers
	* Support for new quantized topologies under brevitas_examples
	* Contributions to the documentation and Jupyter notebooks/tutorials
 	* Bugfixes


2. Submitting your pull request:

	1. Fork this repository to your own GitHub account using the *fork* button above.

	2. Clone the fork to your local computer using *git clone*. Checkout the branch you want to work on.

	3. Please install <a href="https://pre-commit.com/" target="_blank">pre-commit</a> to ensure your code is formatted to our style guidelines.

	4. Add your contribution as needed.

	5. Use *git add*, *git commit*, *git push* to add changes to your fork.

	6. If you are introducing new functionality or fixing a bug, add at least one unit test under the `tests/` folder and make sure it passes before you submit the pull request.

	7. Submit a pull request by clicking the *pull request* button on your GitHub repo:
		* The <a href="https://github.com/Xilinx/brevitas" target="_blank">main branch</a> should always be treated as stable and clean. Only hot fixes are allowed to be pull-requested. The hot fix is supposed to be very important such that without this fix, a lot of things will break.
		*  For new features, bug fixes, doc updates, users should pull request against the <a href="https://github.com/Xilinx/brevitas/tree/dev" target="_blank">development branch</a>.

3. Sign Your Work

Please use the *Signed-off-by* line at the end of your patch which indicates that you accept the Developer Certificate of Origin (DCO) defined by https://developercertificate.org/ reproduced below::

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
```

You can enable Signed-off-by automatically by adding the `-s` flag to the `git commit` command.

Here is an example Signed-off-by line which indicates that the contributor accepts DCO:

```
  This is my commit message

  Signed-off-by: Jane Doe <jane.doe@example.com>
```

4. We will review your contribution and, if any additional fixes or modifications are
necessary, may provide feedback to guide you. When accepted, your pull request will
be merged to the repository. If you have more questions please contact us.
