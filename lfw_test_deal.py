# -*- coding: utf-8 -*-
'''
@authot:李华清
@brief：根据lfw网站所给的图片（lfw，lfwcrop_grey,lfwcrop_color)和pairs.txt,写出需要比较的两幅图像的路径(left.txt,right.txt)以及人脸匹配信息(label.txt)
@brief:选择性进行图像尺度变换和灰度化
'''
import os
import numpy as np
import cv2



#得到一个文件夹下的子文件夹
def dir_subdir(dirname):
    dirname=str(dirname)
    if dirname=="":
        return []
    if dirname[-1]!="/":
        dirname=dirname+"/"
    dirlist = os.listdir(dirname)
    subdir_list=[];
    for x in dirlist:
        if os.path.isdir(dirname+x):
            subdir_list.append(x)
    return subdir_list

#得到一个文件夹下，指定后缀名的文件
def dir_file(dirname,ext):
    dirname=str(dirname)
    if dirname=="":
        return []
    if dirname[-1]!="/":
        dirname=dirname+"/"
    dirlist=os.listdir(dirname)
    filelist=[]
    for x in dirlist:
        if not('.' in x):
            continue
        if x.split('.')[-1]==ext:
            filelist.append(x)
    return filelist

#读取LFW的pairs.txt保存到result中
#result是列表，paris是字典，字典元素flag，img1，img2，num1，num2
def read_paris(filelist="pairs.txt"):
    filelist=str(filelist)
    fp=open(filelist,'r')
    result=[]
    for lines in fp.readlines():
        lines=lines.replace("\n","").split("\t")
        if len(lines)==2:
            print "lenth=2:"+str(lines)
            continue
        elif len(lines)==3:
            pairs={
                    'flag':1,
                    'img1':lines[0],
                    'img2':lines[0],
                    'num1':lines[1],
                    'num2':lines[2],
                    }
            result.append(pairs)
            continue
        elif len(lines)==4:
            pairs={
                'flag':2,
                'img1':lines[0],
                'num1':lines[1],
                'img2':lines[2],
                'num2':lines[3],
                }
            result.append(pairs)
        else:
            print "read file Error!"
            exit()
    fp.close
    print "Read paris.txt DONE!"
    return result


#找到文件夹下后缀名为ext的文件,可包含子文件夹
def dirdir_list(lfwdir,ext='jpg'):
    subdirlist=dir_subdir(lfwdir)
    filelist=[]
    #含有子文件夹
    if len(subdirlist):
        for i in subdirlist:
            list=dir_file(lfwdir+i,ext)
            for j in list:
                j=lfwdir+i+'/'+j
                filelist.append(j)
    #不包含子文件夹
    else:
        list=dir_file(lfwdir,ext)
        for line in list:
            line=lfwdir+line
            filelist.append(line)
    return filelist


def grey_resize(lfwdir,filelist,desdir='lfw_change/',grey=False,resize=False,height=64,width=64):
    if (grey==False and resize==False):
        return []
    #创建目的文件夹
    if not os.path.exists(desdir):
        os.makedirs(desdir)

    if desdir[-1]!='/':
        desdir=desdir+'/'

    #判断文件夹下是否含有子文件夹
    flag=0
    subdir=dir_subdir(lfwdir)
    if len(subdir):
        flag=1

    #创建文件夹
    if flag==1:
        for path in filelist:
            path=path.split('/')
            if not os.path.exists(desdir+path[-2]):
                os.makedirs(desdir+path[-2])

    #处理后图像所放路径
    filelistnew=[]
    for path in filelist:
        path=path.split('/')
        if flag==1:
            filelistnew.append(desdir+path[-2]+'/'+path[-1])
        else:
            filelistnew.append(desdir+path[-1])
    #进行灰度化和尺度变换
    
    num=0
    nums=len(filelistnew)
    for line in filelist:
        img=cv2.imread(line)
       
        if grey==True:
            if num%100==0:
                print "grey:"+str(num)+'/'+str(nums)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if resize==True:
            if num%100==0:
                print "resize:"+str(num)+'/'+str(nums)
            img=cv2.resize(img,(height,width))
        cv2.imwrite(filelistnew[num],img)
        num=num+1
    return filelistnew

#根据pairs.list,灰度化或者统一图像尺度后，得到left.txt,right.txt,label.txt
def split_pairs(pairslist,lfwdir,ext='jpg'):
    num=0
    sum=len(pairslist)

    #lfw图像组织形式，只有图像：0，文件夹+图像：1
    flag=0
    subdir=dir_subdir(lfwdir)
    if len(subdir):
        flag=1

    path_left=""
    path_right=""
    label=""


    #left.txt and right.txt
    #文件夹+图像形式
    print "split pairs.txt"
    if flag==1:
        for lines in pairslist:
            num=num+1
            if num%100==0:
                print str(num)+"/"+str(sum)

            dir_left=lfwdir+lines['img1']+'/'
            dir_right=lfwdir+lines['img2']+'/'
            
            file_left=lines['img1']+'_'+str("%04d" % int(lines["num1"]))+'.'+ext
            file_right=lines['img2']+'_'+str("%04d" % int(lines["num2"]))+'.'+ext
            
            path_left=path_left+dir_left+file_left+"\n"
            path_right=path_right+dir_right+file_right+"\n"

    #图像
    else:
        for lines in pairslist:
            num=num+1
            print str(num)+"/"+str(sum)
            path_left=path_left+lfwdir+lines['img1']+'_'+str("%04d" % int(lines["num1"]))+'.'+ext+"\n"
            path_right=path_right+lfwdir+lines['img2']+'_'+str("%04d" % int(lines["num2"]))+'.'+ext+"\n"
    #label.txt
    for lines in pairslist:    
        if int(lines["flag"])==1:
            label=label+"0\n"
        else:
            label=label+"1\n"
        
    result={
        'path_left':path_left,
        'path_right':path_right,
        'label':label}
    return result


def testdeal(pairs='pairs.txt',lfwdir='lfw/',ext='jpg',changdir='lfw_chang/',grey=False,resize=False,height=64,width=64):
    pairslist=read_paris(pairs)
    if grey==True or resize==True:
        filelist=dirdir_list(lfwdir,ext)
        filelist=grey_resize(lfwdir,filelist,changdir,grey,resize,height,width)
        lfwdir=changdir
    pairs_result=split_pairs(pairslist,lfwdir,ext)
    return pairs_result

def write_pairs(pairs_result,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    left=savepath+'left.txt'
    right=savepath+'right.txt'
    label=savepath+'label.txt'

    fp_left=open(left,'w')
    fp_right=open(right,'w')
    fp_label=open(label,'w')

    fp_left.write(pairs_result['path_left'])
    fp_right.write(pairs_result['path_right'])
    fp_label.write(pairs_result['label'])

    fp_left.close()
    fp_right.close()
    fp_label.close()



def demo_lfw():
    caffe_dir='/home/ikode/caffe-master/'
    pairs=caffe_dir+'data/deepID/pairs.txt'
    lfwdir=caffe_dir+'data/deepID/lfwcrop_color/faces/'
    ext='ppm'
    pairs_result=testdeal(pairs,lfwdir,ext)
    savepath=caffe_dir+'examples/deepID/lfw_'
    write_pairs(pairs_result,savepath)   


def grey_pairs():
    DEEPID='data/deepID_grey/'
    pairs=DEEPID+'pairs.txt'
    lfwdir=DEEPID+'lfwcrop_grey/faces/' 
    ext='pgm'
    pairs_result=testdeal(pairs,lfwdir,ext)
    return pairs_result


    
def demo_webface_resize():
    DEEPID='/home/ikode/caffe-master/data/deepID/'
    pairs=DEEPID+'pairs.txt'
#    lfwdir=DEEPID+'lfwcrop_color/faces/'
    lfwdir='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/croped/'
    ext='jpg'
    lfw_chang='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/change/'

    pairs_result=testdeal(pairs,lfwdir,ext,lfw_chang,False,True)
    savepath=caffe_dir+'examples/deepID/webface_'
    write_pairs(pairs_result,savepath)   



def demo_color_resize():
    DEEPID='/home/ikode/caffe-master/data/deepID/'
    pairs=DEEPID+'pairs.txt'
#    lfwdir=DEEPID+'lfwcrop_color/faces/'
    lfwdir='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/croped/'
    ext='jpg'
    lfw_chang='/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/change/'

    pairs_result=testdeal(pairs,lfwdir,ext,lfw_chang,False,True)
    return pairs_result
    

if __name__=='__main__':
    
    demo_lfw()
