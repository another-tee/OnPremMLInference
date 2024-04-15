# Best Practice:
1. Push the docker images to Azure Container Registry (In this test -> Use offline docker image)
2. You should tag each image:
    - docker tag ml-infer:latest ml-infer:dev
    - docker tag ml-infer:latest ml-infer:uat
    - docker tag ml-infer:latest ml-infer:prd
3. Test