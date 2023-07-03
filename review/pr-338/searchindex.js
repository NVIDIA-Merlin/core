Search.setIndex({docnames:["README","api/index","api/merlin.dag","api/merlin.io","api/merlin.schema","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["README.md","api/index.rst","api/merlin.dag.rst","api/merlin.io.rst","api/merlin.schema.rst","index.rst"],objects:{"":[[1,0,0,"-","merlin"]],"merlin.dag":[[2,1,1,"","BaseOperator"],[2,1,1,"","ColumnSelector"],[2,1,1,"","Graph"],[2,1,1,"","Node"]],"merlin.dag.BaseOperator":[[2,2,1,"","column_mapping"],[2,2,1,"","compute_column_schema"],[2,2,1,"","compute_input_schema"],[2,2,1,"","compute_output_schema"],[2,2,1,"","compute_selector"],[2,2,1,"","create_node"],[2,3,1,"","dependencies"],[2,3,1,"","dynamic_dtypes"],[2,3,1,"","is_subgraph"],[2,3,1,"","label"],[2,2,1,"","load_artifacts"],[2,2,1,"","output_column_names"],[2,3,1,"","output_dtype"],[2,3,1,"","output_properties"],[2,3,1,"","output_tags"],[2,2,1,"","save_artifacts"],[2,3,1,"","supported_formats"],[2,3,1,"","supports"],[2,2,1,"","transform"],[2,2,1,"","validate_schemas"]],"merlin.dag.ColumnSelector":[[2,3,1,"","all"],[2,2,1,"","filter_columns"],[2,3,1,"","grouped_names"],[2,3,1,"","names"],[2,2,1,"","resolve"],[2,3,1,"","tags"]],"merlin.dag.Graph":[[2,2,1,"","clear_stats"],[2,3,1,"","column_mapping"],[2,2,1,"","construct_schema"],[2,2,1,"","get_nodes_by_op_type"],[2,3,1,"","input_dtypes"],[2,3,1,"","input_schema"],[2,3,1,"","leaf_nodes"],[2,3,1,"","output_dtypes"],[2,3,1,"","output_schema"],[2,2,1,"","remove_inputs"],[2,2,1,"","subgraph"]],"merlin.dag.Node":[[2,2,1,"","add_child"],[2,2,1,"","add_dependency"],[2,2,1,"","add_parent"],[2,3,1,"","column_mapping"],[2,2,1,"","compute_schemas"],[2,2,1,"","construct_from"],[2,3,1,"","dependency_columns"],[2,2,1,"","exportable"],[2,3,1,"","graph"],[2,3,1,"","grouped_parents_with_dependencies"],[2,3,1,"","input_columns"],[2,3,1,"","label"],[2,3,1,"","output_columns"],[2,3,1,"","parents_with_dependencies"],[2,2,1,"","remove_child"],[2,2,1,"","remove_inputs"],[2,3,1,"","selector"],[2,2,1,"","validate_schemas"]],"merlin.io":[[3,1,1,"","Dataset"]],"merlin.io.Dataset":[[3,2,1,"","compute"],[3,3,1,"","file_partition_map"],[3,2,1,"","head"],[3,2,1,"","infer_schema"],[3,2,1,"","merge"],[3,3,1,"","npartitions"],[3,3,1,"","num_rows"],[3,3,1,"","partition_lens"],[3,2,1,"","persist"],[3,2,1,"","regenerate_dataset"],[3,2,1,"","repartition"],[3,2,1,"","sample_dtypes"],[3,2,1,"","shuffle_by_keys"],[3,2,1,"","tail"],[3,2,1,"","to_cpu"],[3,2,1,"","to_ddf"],[3,2,1,"","to_gpu"],[3,2,1,"","to_hugectr"],[3,2,1,"","to_iter"],[3,2,1,"","to_npy"],[3,2,1,"","to_parquet"],[3,2,1,"","validate_dataset"]],"merlin.schema":[[4,1,1,"","ColumnSchema"],[4,1,1,"","Schema"],[4,1,1,"","Tags"]],"merlin.schema.ColumnSchema":[[4,4,1,"","dims"],[4,4,1,"","dtype"],[4,3,1,"","float_domain"],[4,3,1,"","int_domain"],[4,4,1,"","is_list"],[4,4,1,"","is_ragged"],[4,4,1,"","name"],[4,4,1,"","properties"],[4,3,1,"","shape"],[4,4,1,"","tags"],[4,3,1,"","value_count"],[4,2,1,"","with_dtype"],[4,2,1,"","with_name"],[4,2,1,"","with_properties"],[4,2,1,"","with_shape"],[4,2,1,"","with_tags"]],"merlin.schema.Schema":[[4,2,1,"","apply"],[4,2,1,"","apply_inverse"],[4,3,1,"","column_names"],[4,2,1,"","copy"],[4,2,1,"","excluding"],[4,2,1,"","excluding_by_name"],[4,2,1,"","excluding_by_tag"],[4,3,1,"","first"],[4,2,1,"","get"],[4,2,1,"","remove_by_tag"],[4,2,1,"","remove_col"],[4,2,1,"","select"],[4,2,1,"","select_by_name"],[4,2,1,"","select_by_tag"],[4,2,1,"","to_pandas"],[4,2,1,"","without"]],"merlin.schema.Tags":[[4,4,1,"","BINARY"],[4,4,1,"","BINARY_CLASSIFICATION"],[4,4,1,"","CATEGORICAL"],[4,4,1,"","CLASSIFICATION"],[4,4,1,"","CONTEXT"],[4,4,1,"","CONTINUOUS"],[4,4,1,"","EMBEDDING"],[4,4,1,"","ID"],[4,4,1,"","ITEM"],[4,4,1,"","ITEM_ID"],[4,4,1,"","LIST"],[4,4,1,"","MULTI_CLASS"],[4,4,1,"","MULTI_CLASS_CLASSIFICATION"],[4,4,1,"","REGRESSION"],[4,4,1,"","SEQUENCE"],[4,4,1,"","SESSION"],[4,4,1,"","SESSION_ID"],[4,4,1,"","TARGET"],[4,4,1,"","TEXT"],[4,4,1,"","TEXT_TOKENIZED"],[4,4,1,"","TIME"],[4,4,1,"","TOKENIZED"],[4,4,1,"","USER"],[4,4,1,"","USER_ID"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:attribute"},terms:{"0":3,"1":3,"10":3,"10mib":3,"11":0,"12":3,"125":3,"128mb":3,"1gb":3,"1mib":3,"2":0,"3":0,"5":3,"5000":3,"5mb":3,"7":0,"abstract":3,"boolean":2,"byte":3,"case":[2,3,4],"class":[0,2,3,4,5],"default":[2,3,4],"do":3,"enum":[3,4],"export":2,"final":3,"float":3,"function":[2,3,4],"int":3,"new":[2,3,4],"return":[2,3,4],"true":[2,3,4],"try":3,"while":3,A:[2,3,4],As:0,By:3,For:[0,3],If:[2,3,4],In:3,NOT:3,That:3,The:[0,2,3,4,5],Then:3,To:[0,3,5],_hugectr:3,_type_:4,about:[0,5],accomplish:3,actual:3,ad:2,add_child:2,add_depend:2,add_metadata_fil:3,add_par:2,addit:3,after:[2,3,4],aggregate_fil:3,algorithm:3,all:[2,3,4],allow:3,alreadi:[2,3],also:[2,3],altern:[2,3],alwai:3,an:[0,2,3],ani:[2,3,4],annotate_list:3,anon:3,anoth:2,api:3,append:3,appli:[2,4],apply_invers:4,approach:3,appropri:3,ar:[0,2,3,4],aren:2,arg:3,argument:3,around:3,array_lik:2,artifact:2,artifact_path:2,assert:3,attribut:3,auto:3,avail:[3,5],avro:3,awai:3,back:3,backend:[2,3],balanc:3,base:[2,3,4],base_dataset:3,base_oper:2,baseoper:2,becaus:3,been:2,befor:[2,3],behavior:3,being:3,best:3,bigfil:3,binari:4,binary_classif:4,bool:[2,3,4],both:3,bring:5,bucket:3,bug:0,build:0,bytesio:3,c:0,cach:3,calcul:[2,3],call:3,can:[2,3,4],cannot:[2,3],capac:3,cat:3,catalog:0,categor:[3,4],chang:3,check:2,child:2,chosen:3,classif:4,classmethod:[2,3],clear:2,clear_stat:2,client:3,cloud:[0,3],cluster:3,col_nam:[2,4],col_selector:2,collect:[3,4],column:[2,3,4],column_map:2,column_nam:4,column_schema:4,columnschema:4,columnselector:[2,4],com:5,combin:[2,3],come:2,complex:3,compon:0,compos:2,comput:[2,3],compute_column_schema:2,compute_input_schema:2,compute_output_schema:2,compute_schema:2,compute_selector:2,concurr:3,consid:2,construct:3,construct_from:2,construct_schema:2,consum:2,cont:3,contain:[0,2,3,4],content:2,context:4,continu:[3,4],contrast:3,control:3,convers:3,convert:[2,3,4],copi:4,core:[2,4],correspond:[2,3],count:3,cpu:3,creat:[3,4],create_nod:2,criteo:3,criteria:3,cross:2,csv:3,cudatoolkit:0,cudf:[2,3],current:[2,3],custom:[2,3],dag:1,dask:3,dask_cudf:3,data:[0,2,3],data_pq:3,dataclass:4,dataformat:2,datafram:[2,3,4],dataframeit:3,dataload:3,dataset:[0,2,3,4],datasetengin:3,datatyp:3,ddf:3,decid:4,dedic:3,defin:[2,3],delai:3,demand:3,dep:2,depend:2,dependencies_selector:2,dependency_column:2,deps_schema:2,deriv:[2,3],describ:[2,4],desir:3,detect:3,determin:[2,3],develop:5,devic:3,dict:[2,3,4],dictarrai:2,dictionari:3,differ:[3,4],dim:4,directli:3,directori:3,disk:[2,3],distinct:3,distinctli:3,distribut:3,document:[0,3,5],doe:[2,3],doesn:2,domain:4,don:[2,4],done:3,dot:3,down:[2,3,4],ds_1:3,ds_2:3,ds_merg:3,dtype:[2,3,4],dynamic_dtyp:2,e:3,each:[2,3,4],ecosystem:4,effici:3,either:[3,4],element:2,embed:4,enabl:[2,3],encod:3,engin:3,epoch:3,error:2,estim:3,exampl:[0,3],except:2,exclud:4,excluding_by_nam:4,excluding_by_tag:4,execut:3,exist:3,expect:2,experiment:3,extens:3,extern:3,extra:2,factori:4,fals:[2,3,4],featur:[2,3],feed:2,file:3,file_0:3,file_1:3,file_min_s:3,file_partition_map:3,file_s:3,fill:2,filter:[2,3],filter_column:2,first:[3,4],flat:3,float_domain:4,follow:3,foo:3,forg:0,format:3,found:[2,4],fraction:3,frame:4,from:[2,3,4],fsspec:3,full:3,fundament:0,further:3,futur:3,g:3,gd:3,gdf:3,gener:3,get:[0,3,4],get_nodes_by_op_typ:2,github:[0,5],given:2,global:3,gpu:[0,3],graph:[2,3],group:[2,3],grouped_nam:2,grouped_parents_with_depend:2,ha:3,handl:3,have:[2,4],head:3,help:0,hive:3,hive_data:3,hook:2,host:3,how:[2,3],http:5,hugectr:3,id:4,ideal:3,identifi:3,ignor:3,ignore_index:3,imag:0,implement:[2,3],improv:3,includ:[0,3],indic:[2,3,4],infer:3,infer_schema:3,inform:[0,3,5],ingest:3,initi:3,initvar:4,inner:3,input:[2,3],input_col:2,input_column:2,input_dtyp:2,input_schema:2,insid:2,inspect:3,instal:3,instanc:3,instead:2,int_domain:4,integ:3,interfac:3,intermedi:3,intern:3,introduct:5,io:[0,1],is_list:4,is_rag:4,is_subgraph:2,isn:3,issu:0,item:4,item_id:4,iter:[3,4],just:4,keep:[2,3],kei:[0,3],keyset:3,kind:2,known:3,kwarg:3,label:[2,3],larg:3,larger:3,latter:3,leaf:2,leaf_nod:2,learn:5,least:2,left:3,length:4,less:3,let:2,level:3,leverag:3,librari:[0,3,5],like:[0,2,3],list:[2,3,4],liter:3,load:[2,3],load_artifact:2,local:3,logic:[2,3],look:[2,3],mai:[2,3],main:[2,3],make:2,map:[2,3],match:[2,3,4],materi:3,matter:2,max:3,maximum:3,mean:3,meet:3,memori:3,merg:3,merin:0,metadata:[3,4],method:[2,3],minim:3,mode:3,model:0,modifi:3,more:[3,5],most:2,move:3,multi:3,multi_class:4,multi_class_classif:4,multipli:3,must:[2,3,4],n:3,name:[0,2,3,4],narrow:2,necessari:3,nest:[2,3],nodabl:2,node:2,non:4,none:[2,3,4],notabl:3,note:3,notebook:3,noth:3,now:3,np:4,npartit:3,npy:3,num_row:3,num_thread:3,numba:0,number:3,nvidia:[0,5],nvtabular:[0,3],object:[2,3,4],occur:2,old:3,one:2,onli:[3,4],op:3,op_typ:2,open:0,oper:[2,3],optim:3,option:[2,3,4],orc:3,order:3,origin:3,os:2,other:[2,3],other_selector:2,otherwis:3,our:5,out:3,out_files_per_proc:3,out_path:3,output:[2,3],output_column:2,output_column_nam:2,output_dtyp:2,output_fil:3,output_files_per_proc:3,output_format:3,output_nod:2,output_path:3,output_properti:2,output_schema:2,output_tag:2,over:3,overarch:5,overrid:[2,3],packag:1,page:0,panda:[2,3,4],parallel:3,paramet:[2,3,4],parent:2,parents_schema:2,parents_selector:2,parents_with_depend:2,parquet:3,parquetdatasetengin:3,part:2,part_mem_fract:3,part_siz:3,partit:3,partition_len:3,partition_on:3,partition_s:3,pass:3,path:[2,3],path_or_sourc:3,pathlik:2,pd:[3,4],per:3,per_partit:3,per_work:3,percent:3,perform:3,persist:3,plan:3,pleas:0,possibl:3,pre:3,preced:3,pred_fn:4,predic:4,prefer:3,prepend:3,present:[2,4],preserv:3,preserve_dtyp:2,preserve_fil:3,pressur:3,prev_output_schema:2,priorit:3,procedur:3,process:3,produc:[2,3],product:3,prohibit:3,project:5,propag:2,properti:[2,3,4],protocol:[2,3],provid:[0,2,4,5],purpos:3,pyarrow:3,python:0,rais:[2,4],random:3,randomli:3,rapidsai:0,raw:3,re:3,read:3,read_parquet:3,real:3,receiv:4,recommend:[0,3],refer:[0,3],regener:3,regenerate_dataset:3,regress:4,rel:3,relat:3,reli:2,reliabl:3,reload:2,remot:3,remov:[2,4],remove_by_tag:4,remove_child:2,remove_col:4,remove_input:2,repartit:3,replac:3,repo:0,report:0,repositori:5,repres:2,represent:2,requir:[2,3],require_metadata_fil:3,resolv:2,result:3,retriev:4,revis:2,right:3,root:3,root_schema:2,roughli:3,row:[3,4],row_group_max_s:3,row_group_s:3,run:2,s3:3,s:[2,3],same:[2,3],sampl:3,sample_dtyp:3,save:2,save_artifact:2,schema:[0,1,2,3],see:[3,5],seed:3,segment:3,select:[2,4],select_by_nam:4,select_by_tag:4,selector:[2,4],sequenc:4,session:4,session_id:4,set:[2,3,4],shape:4,shift:2,should:[2,3,4],shuffl:3,shuffle_by_kei:3,side:3,signific:3,sinc:3,singl:[3,4],size:3,slow:3,smaller:3,so:3,some:3,sort:3,sort_valu:3,sourc:[2,3,4],special:3,specif:3,specifi:3,split:3,stage:3,standard:4,start:[2,5],state:2,statist:2,statoper:2,std:3,still:3,storag:3,storage_opt:3,store:3,str:[2,3,4],straightforward:3,stream:3,strict:2,strict_dtyp:2,string:[2,3],structur:3,sub:2,subgraph:[2,3],subgroup:2,subset:3,suffix:3,suppli:[2,4],support:[2,3],supported_format:2,syntax:2,system:[0,3],t:[2,3,4],tabl:3,tag:[2,4],tagset:4,tail:3,take:[2,3],target:4,task:3,text:4,text_token:4,than:3,thei:2,them:2,therefor:3,thi:[2,3,4],those:3,thread:3,through:3,time:[3,4],tip:3,to_cpu:3,to_ddf:3,to_gpu:3,to_hugectr:3,to_it:3,to_npi:3,to_panda:4,to_parquet:3,to_remov:2,togeth:[2,5],token:4,total:3,transform:[2,3],transformers4rec:0,trickl:2,tupl:[2,4],twice:3,two:3,type:[2,3,4],typeerror:[2,4],underli:3,uniform:3,union:[2,4],uniqu:3,univers:3,unless:3,up:2,upstream:2,url:0,us:[2,3,4],usag:3,use_file_metadata:3,use_ssl:3,user:[3,4],user_id:4,user_rank:3,util:[0,5],valid:[2,3],validate_dataset:3,validate_schema:2,valu:[3,4],value_count:4,valueerror:[2,4],vari:4,varieti:3,veri:3,via:4,wa:3,want:[2,4],warn:3,we:[2,3],websit:5,were:2,what:2,when:[2,3],where:[2,4],whether:[2,3,4],which:[2,3,4],with_dtyp:4,with_nam:4,with_properti:4,with_shap:4,with_tag:4,within:3,without:4,word:3,work:[0,2],worker:3,workflow:[2,3],would:2,wrapper:3,write:3,write_hugectr_keyset:3,written:3,yet:3,you:[2,3,4],your:2},titles:["Merlin Core","merlin namespace","merlin.dag package","merlin.io package","merlin.schema package","Merlin Core"],titleterms:{conda:0,core:[0,5],dag:2,docker:0,feedback:0,index:5,instal:0,io:3,merlin:[0,1,2,3,4,5],namespac:1,packag:[2,3,4],pip:0,relat:5,resourc:5,run:0,schema:4,subpackag:1,support:0,us:0}})