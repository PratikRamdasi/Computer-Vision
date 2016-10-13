import cv2
import glob

def sort_files(path):
    fname=[]
    path= path + "/*.png"
    for file in sorted(glob.glob(path)):
        s=file.split('/')
        a=s[-1].split('\\')
        x=a[-1].split('.')
        literalOne = '('
        literalTwo = ')'
        s= x[0].split(literalOne)[-1].split(literalTwo)[0]
        fname.append(int(s))
    return(sorted(fname))
    
def sort_files_fr(path1):
    fname=[]
    path1 = path1 + "/*.jpg"
    for file in sorted(glob.glob(path1)):
        s=file.split('/')
        a=s[-1].split('\\')
        x=a[-1].split('.')
        literalOne = '('
        literalTwo = ')'
        s= x[0].split(literalOne)[-1].split(literalTwo)[0]
        fname.append(int(s))
    return(sorted(fname))    

def processContoursinGt(path):
    folder = sort_files(path)
    length_cont_gt = []
    for i in range(20,len(folder)):
        newPath = path + "/0 (" + str(i) + ")" + ".png"
        img = cv2.imread(newPath,cv2.CV_LOAD_IMAGE_COLOR)
	#print "Image in processContour: ",img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (31, 31), 0)
        thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        length_cont_gt.append(len(cnts))
    return length_cont_gt    


def processContoursinFr(path1):
    folder1 = sort_files_fr(path1)
    length_cont_fr = []
    for i in range(1,len(folder1)):
        newPath = path1 + "/" + str(i)+ ".jpg"
        img = cv2.imread(newPath,cv2.CV_LOAD_IMAGE_COLOR)
	#print "Image in processContourinFr: ",img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (31, 31), 0)
        thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        length_cont_fr.append(len(cnts))     
    return length_cont_fr

def findPercentAccuracy(gt, fr):
    true_count = 0
    false_count = 0
    for i in range(1,len(gt)):
        if (gt[i] == fr[i]):
            true_count += 1
        else:
            false_count += 1
    return true_count, false_count
            
 
if __name__ == "__main__":
    path = 'groundtruth'
    path1 = 'Frames'
    gt = processContoursinGt(path)
    fr = processContoursinFr(path1)
    true, false = findPercentAccuracy(gt,fr)
    print "Percentage accuracy of objects detected is: {}" .format(float(true)* 100/float(true + false))
    
    
    
