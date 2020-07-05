package main

import (
	"context"
	"fmt"

	vision "cloud.google.com/go/vision/apiv1"
)

// Annotate an image file based on Cloud Vision API, return score and error if exists.
func annotate(uri string) (float32, error) {
	ctx := context.Background()
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		return 0.0, err
	}
	defer client.Close()

	//read img from object uri
	image := vision.NewImageFromURI(uri)

	//face detection process
	annotations, err := client.DetectFaces(ctx, image, nil, 1)
	if err != nil {
		return 0.0, err
	}
	if len(annotations) == 0 {
		fmt.Println("No faces found.")
		return 0.0, nil
	}

	//the returned response is an array, return the 0th element from array
	return annotations[0].DetectionConfidence, nil
}
