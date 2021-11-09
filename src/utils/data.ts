/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

 import * as tf from '@tensorflow/tfjs';

 const IMAGE_SIZE = 784;
 const NUM_CLASSES = 10;
 const NUM_DATASET_ELEMENTS = 65000;

 const TRAIN_TEST_RATIO = 5 / 6;

 const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
 const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

 const MNIST_IMAGES_SPRITE_PATH =
     'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
 const MNIST_LABELS_PATH =
     'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

 /**
  * A class that fetches the sprited MNIST dataset and returns shuffled batches.
  *
  * NOTE: This will get much easier. For now, we do data fetching and
  * manipulation manually.
  */
 export class MnistData {
   constructor() {
     (this as any).shuffledTrainIndex = 0;
     (this as any).shuffledTestIndex = 0;
   }

   async load() {
     // Make a request for the MNIST sprited image.
     const img = new Image();
     const canvas = document.createElement('canvas');
     const ctx = canvas.getContext('2d');
     const imgRequest = new Promise((resolve) => {
       img.crossOrigin = '';
       img.onload = () => {
         img.width = img.naturalWidth;
         img.height = img.naturalHeight;

         const datasetBytesBuffer =
             new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

         const chunkSize = 5000;
         canvas.width = img.width;
         canvas.height = chunkSize;

         for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
           const datasetBytesView = new Float32Array(
               datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
               IMAGE_SIZE * chunkSize);
           ctx?.drawImage(
               img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
               chunkSize);

           const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);

           for (let j = 0; j < (imageData?.data.length || 0) / 4; j++) {
             // All channels hold an equal value since the image is grayscale, so
             // just read the red channel.
             datasetBytesView[j] = (imageData?.data[j * 4] || 0) / 255;
           }
         }
         (this as any).datasetImages = new Float32Array(datasetBytesBuffer);

         resolve(true);
       };
       img.src = MNIST_IMAGES_SPRITE_PATH;
     });

     const labelsRequest = fetch(MNIST_LABELS_PATH);
     const [, labelsResponse] =
         await Promise.all([imgRequest, labelsRequest]);

     (this as any).datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

     // Create shuffled indices into the train/test set for when we select a
     // random dataset element for training / validation.
     (this as any).trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
     (this as any).testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

     // Slice the the images and labels into train and test sets.
     (this as any).trainImages =
         (this as any).datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
     (this as any).testImages = (this as any).datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
     (this as any).trainLabels =
         (this as any).datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
     (this as any).testLabels =
         (this as any).datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
   }

   nextTrainBatch(batchSize: any) {
     return (this as any).nextBatch(
         batchSize, [(this as any).trainImages, (this as any).trainLabels], () => {
           (this as any).shuffledTrainIndex =
               ((this as any).shuffledTrainIndex + 1) % (this as any).trainIndices.length;
           return (this as any).trainIndices[(this as any).shuffledTrainIndex];
         });
   }

   nextTestBatch(batchSize: any) {
     return (this as any).nextBatch(batchSize, [(this as any).testImages, (this as any).testLabels], () => {
       (this as any).shuffledTestIndex =
           ((this as any).shuffledTestIndex + 1) % (this as any).testIndices.length;
       return (this as any).testIndices[(this as any).shuffledTestIndex];
     });
   }

   nextBatch(batchSize: any, data: any, index: any) {
     const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
     const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

     for (let i = 0; i < batchSize; i++) {
       const idx = index();

       const image =
           data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
       batchImagesArray.set(image, i * IMAGE_SIZE);

       const label =
           data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
       batchLabelsArray.set(label, i * NUM_CLASSES);
     }

     const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
     const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

     return {xs, labels};
   }
 }
