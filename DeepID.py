#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Create on Tues 2015-11-24

@author: hqli
'''

#运行前,先将caffe安装好,放在～/caffe-mater
import os


class DeepID():
    #基本属性
    prj=''#项目名称
    caffepath=''#caffe主目录路径
    prjpath=''#工程的位置
    datapath=''#数据的路径
    num=0#人数多少
#    ratio=0#训练数:测试
#    max_iter=0#迭代次数(用于lfw验证)
#    snapshot=0#迭代多少次保存一次
    
    data_train=''#用于训练的数据列表
    data_test=''#用于测试的数据列表
    net_proto=''#caffe的Net
    net_proto_model=''

    solver_proto=''#caffe的solver
    solver_proto_model=''
    deploy_proto=''#caffe的deploy

    snapshot_pre=''#训练好的模型的前缀

    lmdb_train=''#训练集的lmdb
    lmdb_test=''#测试集的lmdb
    imgmean=''#图像均值

    netimg=''#net的结构图像

    log_train=''#caffe的训练日志
    log_test=''#caffe的测试日志
    log_create=''#转换lmdb的日志

    #可执行文件
    shcreate=''#将图像和标签转换成lmdb格式
    shimgmean=''#计算图像均值
    shdrawnet=''#画Net结构图
     
    shtrain=''#训练
    shtest=''#测试


    def __init__(self,prj,caffepath,prjpath,datapath,num):
        self.prj=prj
        self.caffepath=caffepath
        self.prjpath=self.prjpath
        self.datapath=datapath
        self.num=num
#        self.ratio=ratio
#        self.itera=max_iter
#        self.snapshot=snapshot
       
        #在prjpath下新建名为num的文件夹,将配置以及生成的文件放在该文件夹下
        #在prjpath下新建num文件夹
        if not os.path.exists(prjpath+str(num)):    
            os.makedirs(prjpath+str(num))
        
        self.net_proto_model=prjpath+prj+'_train_test.prototxt'
        self.solver_proto_model=prjpath+prj+'_solver.prototxt'
        
        self.netimg=prjpath+prj+'_net.png'
        self.shdrawnet=prjpath+prj+'_drawnet.sh'#画Net结构图
        
        rspath=prjpath+str(num)+'/'
        strnum='_'+str(num)+'_'
  
        self.data_train=rspath+prj+'_train_'+str(num)+'.txt'
        self.data_test=rspath+prj+'_val_'+str(num)+'.txt'

        self.net_proto=rspath+prj+strnum+'train_test.prototxt'

        self.solver_proto=rspath+prj+strnum+'solver.prototxt'

        self.deploy_proto=rspath+prj+'_deploy.prototxt'

        self.snapshot_pre=rspath+prj+'_'+str(num)
        
        self.lmdb_train=rspath+prj+strnum+'train_lmdb'
        self.lmdb_test=rspath+prj+strnum+'test_lmdb'    
        self.imgmean=rspath+prj+strnum+'mean.binaryproto'


        self.log_train=rspath+prj+strnum+'train.log'
        self.log_test=rspath+prj+strnum+'test.log'
        self.log_create=rspath+prj+strnum+'create.log'

        #可执行文件
        self.shcreate=rspath+prj+strnum+'create.sh'#将图像和标签转换成lmdb格式
        self.shimgmean=rspath+prj+strnum+'imgmean.sh'#计算图像均值
 
        self.shtrain=rspath+prj+strnum+'train.sh'#训练deepID
        self.shtest=rspath+prj+strnum+'test.sh'#测试deepID

        
    @staticmethod
    def fileopt(filename,content):
        fp=open(filename,'w')
        fp.write(content)
        fp.close()

 
    def div_data(self,ratio):
        '''
        @brief: 将数据集分为训练集和测试集,保存在prjpath下
        '''

#        if not os.path.exists(savepath):
#            os.makedirs(savepath)

        dirlists=os.listdir(self.datapath)
        dict_id_num={}
        for subdir in dirlists:
            dict_id_num[subdir]=len(os.listdir(os.path.join(self.datapath,subdir)))
        #sorted(dict_id_num.items, key=lambda dict_id_num:dict_id_num[1])
        sorted_num_id=sorted([(v, k) for k, v in dict_id_num.items()], reverse=True)
        select_ids=sorted_num_id[0:self.num]
        
        fid_train=open(self.data_train,'w')
        fid_test=open(self.data_test,'w')
        
        pid=0
        for  select_id in select_ids:
            subdir=select_id[1]
            filenamelist=os.listdir(os.path.join(self.datapath,subdir)) 
            num=1
            for filename in filenamelist :
                #print select_ids[top_num-1]
                if num>select_ids[self.num-1][0]:
                    break
                if num%ratio!=0:
                    fid_train.write(os.path.join(subdir,filename)+'\t'+str(pid)+'\n')
                else:
                    fid_test.write(os.path.join(subdir,filename)+'\t'+str(pid)+'\n')
                num=num+1
            pid=pid+1
        fid_train.close()
        fid_test.close()   
    
    def create(self,height,width):
        '''
        @brief : 
        @param : 
        '''
        _command='rm -rf '+self.lmdb_train+'\n'
        _command+='rm -rf '+self.lmdb_train+'\n'    
    
        _command+='echo "Creating train lmdb..."\n'
        
        _command+=self.caffepath+'build/tools/convert_imageset '
        _command+='--resize_height='+str(height)+' '
        _command+='--resize_width='+str(width)+' '
        _command+='--shuffle '
        _command+='--backend="lmdb" '
        _command+=self.datapath+' '
        _command+=self.data_train+' '
        _command+=self.lmdb_train
        out=r' 2>&1 |tee '+self.log_create
        _command+=out+'\n'

        _command+='echo "Creating train lmdb..."\n'
        
        _command+=self.caffepath+'build/tools/convert_imageset '
        _command+='--resize_height='+str(height)+' '
        _command+='--resize_width='+str(width)+' '
        _command+='--shuffle '
        _command+='--backend="lmdb" '
        _command+=self.datapath+' ' 
        _command+=self.data_test+' '
        _command+=self.lmdb_test
        out=r' 2>&1 |tee -a '+self.log_create
        _command+=out+'\n'
    
        _command+='echo "Done"'
    
        DeepID.fileopt(self.shcreate,_command)

        os.system(_command)

    def compute_imgmean(self):
        _command=self.caffepath+'build/tools/compute_image_mean '+self.lmdb_train+' '+self.imgmean
        DeepID.fileopt(self.shimgmean,_command)

        os.system(_command)
    
    def draw_net(self):
        _command='python '+self.caffepath+'python/draw_net.py '+self.net_proto_model+' '+self.netimg
        DeepID.fileopt(self.shdrawnet,_command)

        os.system(_command)
   
    def train(self):
        #将project_train_test.prototxt中lmdb和imgmean替换
        #重新写入到num文件夹下/project_num_train_test.prototxt
        lmdb_train=self.lmdb_train.replace('/','\\/')
        lmdb_test=self.lmdb_test.replace('/','\\/')
        imgmean=self.imgmean.replace('/','\\/')
        
        lmdb_train='    source: \\\"'+lmdb_train+'\\\"'
        lmdb_test='    source: \\\"'+lmdb_test+'\\\"'
        imgmean='    mean_file: \\\"'+imgmean+'\\\"'
        
        print lmdb_train
        num_train=9
        num_train_mean=14
        num_test=26
        num_test_mean=31

        _command='cp '+self.net_proto_model+' '+self.net_proto+'\n'
        _command+='cp '+self.solver_proto_model+' '+self.solver_proto+'\n'
        
        _command+='sed -i -e \"'+str(num_train)+'c\\'+lmdb_train+'\" '+self.net_proto+'\n'
        _command+='sed -i -e \"'+str(num_train_mean)+'c\\'+imgmean+'\" '+self.net_proto+'\n'
        _command+='sed -i -e \"'+str(num_test)+'c\\'+lmdb_test+'\" '+self.net_proto+'\n'
        _command+='sed -i -e \"'+str(num_test_mean)+'c\\'+imgmean+'\" '+self.net_proto+'\n'
       
        num_net=1
        num_snapshot_pre=15
        net_proto=self.net_proto.replace('/','\\/')
        snapshot_pre=self.snapshot_pre.replace('/','\\/')
        
        net_proto='net: \\\"'+net_proto+'\\\"'
        snapshot_pre='snapshot_prefix: \\\"'+snapshot_pre+'\\\"'
        
        _command+='sed -i -e \"'+str(num_net)+'c\\'+net_proto+'\" '+self.solver_proto+'\n'
        _command+='sed -i -e \"'+str(num_snapshot_pre)+'c\\'+snapshot_pre+'\" '+self.solver_proto+'\n'
        
 
        out=r' 2>&1 |tee '
        _command+=self.caffepath+'/build/tools/caffe train '+'--solver='+self.solver_proto+' '+out+self.log_train
        DeepID.fileopt(self.shtrain,_command)
        os.system(_command)
        
    def resume(self,num):
        #将project_train_test.prototxt中lmdb和imgmean替换
        #重新写入到num文件夹下/project_num_train_test.prototxt
        lmdb_train=self.lmdb_train.replace('/','\\/')
        lmdb_test=self.lmdb_test.replace('/','\\/')
        imgmean=self.imgmean.replace('/','\\/')
        
        lmdb_train='    source: \\\"'+lmdb_train+'\\\"'
        lmdb_test='    source: \\\"'+lmdb_test+'\\\"'
        imgmean='    mean_file: \\\"'+imgmean+'\\\"'
        
        print lmdb_train
        num_train=9
        num_train_mean=14
        num_test=26
        num_test_mean=31

        _command='cp '+self.net_proto_model+' '+self.net_proto+'\n'
        _command+='cp '+self.solver_proto_model+' '+self.solver_proto+'\n'
        
        _command+='sed -i -e \"'+str(num_train)+'c\\'+lmdb_train+'\" '+self.net_proto+'\n'
        _command+='sed -i -e \"'+str(num_train_mean)+'c\\'+imgmean+'\" '+self.net_proto+'\n'
        _command+='sed -i -e \"'+str(num_test)+'c\\'+lmdb_test+'\" '+self.net_proto+'\n'
        _command+='sed -i -e \"'+str(num_test_mean)+'c\\'+imgmean+'\" '+self.net_proto+'\n'
       
        num_net=1
        num_snapshot_pre=15
        net_proto=self.net_proto.replace('/','\\/')
        snapshot_pre=self.snapshot_pre.replace('/','\\/')
        
        net_proto='net: \\\"'+net_proto+'\\\"'
        snapshot_pre='snapshot_prefix: \\\"'+snapshot_pre+'\\\"'
        
        _command+='sed -i -e \"'+str(num_net)+'c\\'+net_proto+'\" '+self.solver_proto+'\n'
        _command+='sed -i -e \"'+str(num_snapshot_pre)+'c\\'+snapshot_pre+'\" '+self.solver_proto+'\n'
        
 
        out=r' 2>&1 |tee '
        _command+=self.caffepath+'/build/tools/caffe train '+'--solver='+self.solver_proto+' '+out+self.log_train
        DeepID.fileopt(self.shtrain,_command)
        
    def test(self,iternum):
        out=r' 2>&1 |tee '
        _command=self.caffepath+'/build/tools/caffe test --model='+self.net_proto+' --weights='+self.snapshot_pre+'_iter_'+str(iternum)+'.caffemodel'+out+log_test
        DeepID.fileopt(self.shtest,_command)
        os.system(_command)
def demo(num):
    deepID=DeepID('deepID','/home/ikode/caffe-master/','/home/ikode/caffe-master/examples/deepID/','/media/ikode/Document/big_materials/document/deep_learning/caffe/face_datasets/webface/croped/',num)
    ratio=9

    deepID.div_data(ratio)

    deepID.create(64,64)

    deepID.compute_imgmean()

#    deepID.draw_net()

    deepID.train()

if __name__=='__main__':

    demo(1000)
    #demo后面的数字是训练的人数（1-10575）
