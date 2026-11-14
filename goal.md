# Goal
Goal of this project is to enhance Immich, the self-hostable photo management software, with better person re-identification.
Stock Immich does face detectioon and recognition, but this falls short when people eg. do not face the camera or the face is blurry etc. 

By using Person Re-identification techniques, we can look at other features besides just the face to re-identify (Re-ID) people.

The goal, at least initially, is to run a separate Docker container, that talks with the immich API to perform this Re-ID task.

Face recognition, by Immich itself is still leading in providing ground truth face labels. 
But, that may miss some faces and then a cluster of similar faces may be proposed to have the same face after clustering with Re-ID.
It's crucial that the R-ID process is only done on short timespans, eg within the same day or even a few hours, as people clothes changes on that timespanand re-ID beyond that is less reliable.

So, for example: 
- we have a photo without a face label, but the face/person is part of cluster 'a' as found by re-ID.
- another photo has another person with label 'a' as foun by re-ID. That person is tagged as name "Albert" in that photo. Then, we can (propose to) label all the other people labeled 'a' as "Albert"

I say proposed, because this need human confirmation, at least initially.
Maybe that confirmation can be removed after we gain confidence in the method, or based on a threshold. 

One model that could do this job is OSNet / torchreid: https://kaiyangzhou.github.io/deep-person-reid/index.html

I need a docker container running this model and interacting with the Immich API to propose faces. 
The proposals and confirmation of that goes through a custom web-interface outside Immich. 
Maybe later that we really integrate with Immich directly. 

Make sure to use the same pytorch version as Immich itself uses. 
