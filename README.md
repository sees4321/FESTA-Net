<!-- readme template: https://github.com/othneildrew/Best-README-Template/blob/main/BLANK_README.md -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sees4321/FESTA-Net">
    <img src="images/model_overview.png" alt="Logo" width="5550" height="2273">
  </a>

<h1 align="center">FESTA-Net: A Generalizable Temporal-AlignmentNetwork for Robust Affective and Cognitive StateDecoding from Heterogeneous Neural Signals</h1>

  <p align="center">
    This is pytorch implementation of FESTA-Net
    <br />
    <a href="https://github.com/sees4321/FESTA-Net"><strong>[paper link]</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a> -->
  </p>
</div>


## Abstract

Hybrid functional near-infrared spectroscopy(fNIRS) and electroencephalography (EEG) systems providecomplementary hemodynamic and electrophysiologicalinformation, offering strong potential for passive brain–computer interfaces (pBCIs). Nevertheless, effectively decodingcomplex affective dynamics from heterogeneous neural signalswith distinct temporal characteristics remains a major challenge,particularly when a robust performance across different tasksand channel configurations is required. This study proposesFESTA-Net, a synchronized temporal alignment network thatenables robust brain-state decoding by explicitly preserving thetemporal correspondence between fNIRS and EEG signals. Theproposed approach partitions multimodal signals into equal-length temporal segments, within which modality-specificencoders learn localized representations, while a transformer-based module captures long-range intersegment dependencies.Fusion is performed at the segment level, enabling FESTA-Net tomaintain a temporally synchronized representation ofelectrophysiological and hemodynamic dynamics acrossmodalities. The effectiveness and generalizability of the proposednetwork are validated using four heterogeneous datasetsencompassing emotion recognition, word generation,discrimination–selection response, and N-back tasks, eachinvolving different channel configurations. The experimentalresults demonstrate that FESTA-Net consistently outperformsstate-of-the-art unimodal and hybrid models across all datasets.Extensive ablation studies further confirm the contribution of theproposed temporal-alignment fusion mechanism, whereasadditional analyses indicate improved computational efficiencyand enhanced interpretability. Overall, the results demonstratethat FESTA-Net provides a generalizable and robust multimodaldecoding framework, thereby advancing the practicaldeployment of hybrid pBCI systems in diverse experimental andreal-world settings.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

If you have any question, please contact at minsukim207@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Citations

```bibtex
@article{Kim2026,
    title   = {FESTA-Net: A Generalizable Temporal-AlignmentNetwork for Robust Affective and Cognitive StateDecoding from Heterogeneous Neural Signals},
    author  = {Minsu Kim, Kyeonggu Lee, Minyoung Chun, Hyerin Nam, and Chang-Hwan Im},
    journal = {None},
    year    = {2026},
    pages   = {None},
    url     = {paperlink}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username -->
