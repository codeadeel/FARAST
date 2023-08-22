# Face Analytics Demo Application Clients

The Face system offered to clients serves a multitude of purposes across various demonstration scenarios. These scenarios encompass functions such as facial enrollment and real-time demonstrations for tasks like face detection, recognition, alignment, anti-spoofing, and tracking. These functions are based on the analysis of live camera streams and RTSP feeds.

* [**Live Demo**](#livedemo)
* [**Face Enrollment Demo**](#livenroll)


## <a name="livedemo">Live Demo

The subject [***code***][liveDemo link] assumes the role of conducting a face recognition demonstration using live streams sourced from both webcams and RTSP feeds. This demonstration incorporates adjustable parameters, allowing modifications that directly influence the outcomes and operational efficiency of the system within the designated stream. Presented below is the roster of requisite environment variables crucial for the seamless execution of the live face system demonstration associated with the subject.

```bash
export serverIP='0.0.0.0:4343'                        # Face System Inference IP
export meshModel='/home/faceMesh.task'                # Face Landmarks Model Address
```

## <a name="livenroll">Face Enrollment Demo

The subject [***code***][liveEnroll link] bears the responsibility of acquiring a live stream from a webcam for the purpose of facial enrollment. The facial data obtained from the subject is subsequently stored within a designated bucket storage and promptly employed for real-time inference tasks. Presented below is the compilation of essential environment variables necessary to facilitate the seamless execution of the live face enrollment demonstration associated with the subject.

```bash
export bucketAccessKey=abc                            # MINIO access key
export bucketSecretKey=abc                            # MINIO secret key
export secureBucket=false                             # MINIO bucket https enabled or not
export bucketUrl='http://172.17.0.5:8009'             # MINIO bucket API URL
export facesTemplatesBucket=facebook                  # Face Storing Bucket Name
export serverIP='http://172.17.0.5:8080'              # Face System Inference IP
export submissionCode=abc                             # Face Capture Submission Code
```

[liveDemo link]: ./liveDemo.py
[liveEnroll link]: ./faceEnroll.py
