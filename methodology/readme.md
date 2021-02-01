<h1>Our Algorithm is divided to several steps, I will go through them in this section:</h1>

<h2>1. Deskewing the Image:</h2>
if the image is rotated or skewed by any degree, we first detect lines using hough lines from the canny edges image, then we find maximum line that determine the orientation of the image, then we rotate the image in opposite direction to make that line horizontal thus deskewing any skewed image, and if the image already not skewed then it will rotate by 0 degrees ie do nothing
<h2>2. Convert Image To Binary:</h2>
we convert the image to binary image using local threshold with suitable block size and offset, and then invert it so lines and musical notes are white and background is black
<h2>3. Remove Staff Lines :</h2>
we remove staff lines from the image, by first dividing image into multiple patches and for each patch we calculate staff height and staff space using run length encoding technique and getting most frequent white and black runs, then we loop through each column in the patch and find if the successive run length encoding is more than staff height ie: not staff line we extract the musical note and remove the line
<h2>4. Remove Musical Notes:</h2>
in this step we want to remove musical note from the image and only leave staff lines do we can get position of the note precisely, so we do the same algorithms as the one mentioned above but instead of removing lines we remove the note and then by applying some dilation and erosion steps to connect lines and remove noise we obtain an image with only staff lines and no musical note
<h2>5. Extract Each 5 Staves Together :</h2>
in this step we want to dived the image and do a line segmentation ie: each 5 staff lines will be a one line Like in OCR, so we dilate the original  image with a Structure element that its height is the staff space thus merging each 5 lines together then we find the contours in the image and each contour we find will be a line 
<h2>6. Musical Note Segmentation:</h2>
in this step we want to extract each musical note from the extracted line, we do that by applying label function to our image thus obtaining a label for each connected component and then each connected component will be a musical note on its own
<h2>7. Remove Unwanted labels:</h2>
the output of the last step may produce a lot of false notes and characters thus we make a filter for each image to check if its garbage or a musical note by checking its dimension and contrast
<h2>8. Classify Musical Note:</h2>
in this step we classify out musical note using out hog classifier to see what symbol or note is the image, and depending on the output we either add it to our array or we get the position of this note in line to figure out its value whether its ( c, a, b,...etc)
<h2>9. Convert Line Array To Usable Data:</h2>
in this step we convert the output array from the classifier module to a usable data by merging symbols with suitable notes and merging time stamps together 
<h2>10. Output To File:</h2>
this is the last step where we save our output to a text file following GUIDO rules
