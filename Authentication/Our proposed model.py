
file=open("new file.txt","w")

def encode_file(file_name):
    csv_text = "here is the text of csv file you have written"
    
    # Read lines from the target file
    protocol_no = 0
    for line in open(file_name, encoding='utf-8'):
        encpara= []
        enckey = []
        final=[]
        
        parties=[]
        # Handle each line: just replace("\n", "").split the parameters into a list
        if line.find("Tag") != -1:
             
             file.write('\n')
             protocol_no=protocol_no+1
        
               
        elif line.startswith("Name"):
              pass
        
        elif  line.startswith("Paper") or line.find('→') == -1 or line.startswith("Flaw"):
           #print("protocol {}, {}".format(protocol_no, line))
             pass
                     
        else:

            
            file.write('\n')
            idx = line.find(":")
            entities=line[:idx]
            arrow=entities.find('→')
            if arrow != -1:
               sender=entities[:arrow]
               idA=sender.find('A')
               idB=sender.find('B')
               idS=sender.find('S')
               if idA != -1:
                  parties=['101']
               if idB != -1:
                  parties=['102']
               if idS != -1:
                  parties=['103']

               receiv=entities[arrow+1:]

               idA=receiv.find('A')
               idB=receiv.find('B')
               idS=receiv.find('S')
               if idA != -1:
                  parties=parties+['101']
               if idB != -1:
                  parties=parties+['102']
               if idS != -1:
                  parties=parties+['103']

            body = line[idx + 1:]

           

                  
            # replace("\n", "").splitting hash
            
            idhash=body.find("hash")
            idhashend=body.rfind("}")
            if idhash != -1 :
               hashvalue=body[idhash+5:idhashend]

               idEnchash=hashvalue.find("E_(")
               idsighash=hashvalue.find("Sig")
               if idEnchash != -1 :
                  Enchash=hashvalue[idEnchash:]
                  idkH=hashvalue.find(")")
                  idkHEnd=hashvalue.find("}")
                  #nested encryption key,param
                  KEYHash=hashvalue[idEnchash+3:idkH]
                  paramEH=hashvalue[idkH+3:idkHEnd]

                  final=['4']+['3']+[KEYHash]
                  final = final+paramEH.replace("\n", "").split(",")

                  hashparam1=hashvalue[:idEnchash]
                  hashparam2=hashvalue[idkHEnd+2:idhashend]
                  hashparam=hashparam1+','+hashparam2

                  finalhash = final+hashparam.replace("\n", "").split(",")
                  print(finalhash)
                  
 
                
               elif idsighash != -1 :
                   sighash=hashvalue[idsighash:]
                   idkH=hashvalue.find(")")
                   idkHEnd=hashvalue.find("}")

                  #nested signature key,param
                   sigKEYHash=hashvalue[idsighash+5:idkH]
                   paramsigH=hashvalue[idkH+3:idkHEnd]

                   final=['4']+['2']+[sigKEYHash]
                   final = final+paramsigH.replace("\n", "").split(",")

                   hashparam1=hashvalue[:idsighash]
                   hashparam2=hashvalue[idkHEnd+2:idhashend]
                   hashparam=hashparam1+','+hashparam2

                   finalhash = final+hashparam.replace("\n", "").split(",")

                   
                  

                  

               else:
                    finalhash = ['4']+hashvalue.replace("\n", "").split(",")
                    

                   
                    
                    
                   
                  

               
            # splitting plaintext
               idx = body.find("E_(")
               idSIGN=body.find("Sig_")
               if idx != -1 and idx < idSIGN:
                  
                  plain=body[:idx]
                  
                  
                  finalx = plain.replace("\n", "").split(",")
                  
                 
                  final=parties+['1']+finalx 
                  
                  
                  
                  
               elif idSIGN != -1 and idSIGN < idhash: 
                    plain=body[:idSIGN]
                    finalx = plain.replace("\n", "").split(",")
                  
                   
                    final=parties+['1']+finalx 
                    
                  
                                     
               elif idx != -1 and idx < idhash:
                    plain=body[:idx]
                    finalx = plain.replace("\n", "").split(",")
                  
                    n=8-len(finalx)
                    final=parties+['1']+finalx 
                    
                    
                    

               else:
                    plain=body[:idhash]
                    
                    finalx = plain.replace("\n", "").split(",")
                  
                    n=8-len(finalx)
                    final=parties+['1']+finalx 
                    
                    
                    
                    
            # splitting SIGNATURE
               
               if idSIGN != -1 :
                  sign=body[idSIGN:idhash]
                  idsigkey=sign.find(")")
                  idsign=sign.find("Sig_")  
                  signkey=sign[idsign+5:idsigkey] 
                  final=final+['2']+[signkey]
               
                  idendsign=sign.rfind("}")
                  
                  
                    
                  # nested encryption of sign
                  idnestedEnc=sign.find("E_(")
                  
                  if idnestedEnc != -1:
                    
                     signature1=sign[idsigkey+3:idnestedEnc]
                     
                     idEndnestedEnc=sign.find("}")
                     signature2=sign[idEndnestedEnc+2:idendsign]
                     
                     idEnckey=sign.find(")")
                     nestedEnc=sign[idnestedEnc:]
                     idnestedEnckey=nestedEnc.find(")")
                     nestedEnc=sign[idnestedEnc:idEndnestedEnc]
                     # nested Encryption key,param
                     nestedEnckey=nestedEnc[4:idnestedEnckey]
                     nestedEncparam=nestedEnc[idnestedEnckey+3:]
                     

                     signature=signature1+','+signature2
                     final = final+signature.replace("\n", "").split(",")

                     final=final+[nestedEnckey]
                     final=final+nestedEncparam.replace("\n", "").split(",")
                     n=23-len(final)
                     final=final 
                     
                     
                     
                  else:
                      signature=sign[idsigkey+3:idendsign] 
                      final = final+signature.replace("\n", "").split(",")
                      n=23-len(final)
                      final=final 
                      
                      
               
                    
           
                     

            # splitting encryption

                  encr= body[idx:idSIGN]
                  idk=encr.find(")")
                  idend1=encr.find("}")
            
           
                  if idx != -1 :
                     
                     
                     idxx = encr.find("E_(")
                     KEY1=encr[idxx+3:idk]
                     final=final +['3']+ [KEY1]
                     
               
                     X=encr[idk:idSIGN]
                     idnestsig = encr.find("sig")
                     nestsig=encr[idnestsig:idSIGN]
                     idnestsig = nestsig.find("sig")
                     idnestsigkey=nestsig.find(")")
                     nestsigkey=nestsig[idnestsig+5:idnestsigkey]
                     final = final+[nestsigkey]
                     idendsig=nestsig.find("}")
                     nestsignature=nestsig[idnestsigkey+3:idendsig]
                     final = final+nestsignature.replace("\n", "").split(",")
                     
                    
                  

               
                     idx2=X.find("E_(")
                      
                           
                     if idx2 != -1:
                     
                        idx3=X.find("}")
                        
                        if idx2 > idx3:
                           
                           
                           if idnestsig!= -1:
                              idnestsig = encr.find("sig")
                              encr11=encr[idk+3:idnestsig]
                              
                              XX=X[idx3+1:]
                              idend12=XX.find("}")
                              
                              encr12=XX[1:idend12]
                              encr1=encr11+','+encr12
                              final = final+encr1.replace("\n", "").split(",")
                              idxx2=XX.find("E_(")
                              idk2=XX.find(")")
                              KEY2=XX[idxx2+3:idk2]
                              final = final+['3']+[KEY2]
                              encr2=XX[idk2+3:]
                              idend2=encr2.find("}")
                              encr2=encr2[:idend2]
                              
                              final = final+encr2.replace("\n", "").split(",")
                              n=43-len(final)
                              final=final +finalhash
                              print(final)
                              for element in final:
                                  file.write(element)
                                  file.write(',')
                              
                              
                              
                           else:
                                 encr1=encr[idk+3:idend1]
                                 final = final+encr1.replace("\n", "").split(",")
                                 XX=X[idx3+1:]
                                 idxx2=XX.find("E_(")
                                 idk2=XX.find(")")
                                 KEY2=XX[idxx2+3:idk2]
                                 final = final+['3']+[KEY2]
                                 idend2=XX.find("}")
                                 encr2=XX[idk2+3:idend2]
                                 final = final+encr2.replace("\n", "").split(",")
                                 n=43-len(final)
                                 final=final +finalhash
                                 print(final)
                                 for element in final:
                                     file.write(element)
                                     file.write(',')
                                 
                                 
                           
                          
                           
                        
                        else:
                        
                           idx4=encr.rfind("}")
                           encr1=encr[idk+3:idx4]
                           idx5=encr1.find("E_(")
                           idknest=encr1.find(")")
                           KEYnest=encr1[idx5+3:idknest]
                           final = final+[KEYnest]
                           idend2=encr1.find("}")
                           encrNest=encr1[idknest+3:idend2]
                           final = final+encrNest.replace("\n", "").split(",")
                           n=43-len(final)
                           final=final +finalhash
                           print(final)
                           for element in final:
                               file.write(element)
                               file.write(',')

                     else:
                           encr1=encr[idk+3:idend1]
                           idnestsig = encr.find("sig")
                           encr11=encr[idk+3:idnestsig]
                           idx3=X.find("}")         
                           XX=X[idx3+1:]
                           idend12=XX.find("}")
                                    
                           encr12=XX[1:idend12]
                           encr1=encr11+','+encr12
                           final = final+encr1.replace("\n", "").split(",")
                           n=43-len(final)
                           final=final +finalhash
                           print(final)
                           for element in final:
                               file.write(element)
                               file.write(',')
                  else:
                       final=final +['3']+finalhash
                       print(final)
                       for element in final:
                           file.write(element)
                           file.write(',')



                        
                           

               #when there is no signature        
               else:
                    final=final +['2']
                    idx = body.find("E_(")
                    encr= body[idx:idhash]
                    
            
           
                    if idx != -1 :
                       idk=encr.find(")")
                       idend1=encr.find("}")
                       idxx = encr.find("E_(")
                       KEY1=encr[idxx+3:idk]
                       final=final+['3']+[KEY1]
                           
                       X=encr[idk:idSIGN]
                       idnestsig = encr.find("sig")
                       nestsig=encr[idnestsig:idSIGN]
                       idnestsig = nestsig.find("sig")
                       idnestsigkey=nestsig.find(")")
                       nestsigkey=nestsig[idnestsig+5:idnestsigkey]
                       final = final+[nestsigkey]
                       idendsig=nestsig.find("}")
                       nestsignature=nestsig[idnestsigkey+3:idendsig]
                       final = final+nestsignature.replace("\n", "").split(",")
                       

                     
                  

                  
                       idx2=X.find("E_(")
                        
                              
                       if idx2 != -1:
                        
                          idx3=X.find("}")
                           
                          if idx2 > idx3:
                              
                              
                             if idnestsig!= -1:
                                idnestsig = encr.find("sig")
                                encr11=encr[idk+3:idnestsig]
                                 
                                XX=X[idx3+1:]
                                idend12=XX.find("}")
                                 
                                encr12=XX[1:idend12]
                                encr1=encr11+','+encr12
                                final = final+encr1.replace("\n", "").split(",")
                                idxx2=XX.find("E_(")
                                idk2=XX.find(")")
                                KEY2=XX[idxx2+3:idk2]
                                final = final+['3']+[KEY2]
                                encr2=XX[idk2+3:]
                                idend2=encr2.find("}")
                                encr2=encr2[:idend2]
                                 
                                final = final+encr2.replace("\n", "").split(",")
                                n=43-len(final)
                                final=final +finalhash
                                print(final)
                                for element in final:
                                    file.write(element)
                                    file.write(',')
                                
                              
                                 
                             else:
                                  encr1=encr[idk+3:idend1]
                                  final = final+encr1.replace("\n", "").split(",")
                                  XX=X[idx3+1:]
                                  idxx2=XX.find("E_(")
                                  idk2=XX.find(")")
                                  KEY2=XX[idxx2+3:idk2]
                                  final = final+['3']+[KEY2]
                                  idend2=XX.find("}")
                                  encr2=XX[idk2+3:idend2]
                                  final = final+encr2.replace("\n", "").split(",")
                                  n=43-len(final)
                                  final=final +finalhash
                                  print(final)
                                  for element in final:
                                      file.write(element)
                                      file.write(',')
                                  
                                 
                           
                          
                           
                           
                          else:
                           
                               idx4=encr.rfind("}")
                               encr1=encr[idk+3:idx4]
                               idx5=encr1.find("E_(")
                               idknest=encr1.find(")")
                               KEYnest=encr1[idx5+3:idknest]
                               final = final+[KEYnest]
                               idend2=encr1.find("}")
                               encrNest=encr1[idknest+3:idend2]
                               final = final+encrNest.replace("\n", "").split(",")
                               n=43-len(final)
                               final=final +finalhash
                               print(final)
                               
                               for element in final:
                                    file.write(element)
                                    file.write(',')
                               

                       else:
                            encr1=encr[idk+3:idend1]
                            idnestsig = encr.find("sig")
                            encr11=encr[idk+3:idnestsig]
                            idx3=X.find("}")         
                            XX=X[idx3+1:]
                            idend12=XX.find("}")
                                       
                            encr12=XX[1:idend12]
                            encr1=encr11+','+encr12
                            final = final+encr1.replace("\n", "").split(",")
                            n=43-len(final)
                            final=final +finalhash
                            print(final)
                            for element in final:
                                file.write(element)
                                file.write(',')
                    else:
                           final=final +['3']+finalhash
                           print(final)
                           for element in final:
                               file.write(element)
                               file.write(',')
                            
                            
                           


            #WHEN THERE IS NO HASH
            else:
                 finalhash=['4']

            #  splitting plaintext
                 idx = body.find("E_(")
                 idSIGN=body.find("Sig_")
                 if idx != -1 and idx < idSIGN:
                  
                    plain=body[:idx]
                     
                     
                    finalx = plain.replace("\n", "").split(",")
                     
                    n=8-len(finalx)
                    final=parties+['1']+finalx 
                     
                     
                     
                     
                 elif idSIGN != -1 : 
                      plain=body[:idSIGN]
                      finalx = plain.replace("\n", "").split(",")
                     
                      n=8-len(finalx)
                      final=parties+['1']+finalx 
                     
                     
                                       
                 elif idx != -1 :
                      plain=body[:idx]
                      finalx = plain.replace("\n", "").split(",")
                     
                      n=8-len(finalx)
                      final=parties+['1']+finalx 
                     
                     
                     

                 else:
                      plain=body[:]
                      
                  
                      finalx = plain.replace("\n", "").split(",")
                     
                      n=8-len(finalx)
                      final=parties+['1']+finalx
                      
                     
                     
                     
                     
               # splitting SIGNATURE
                  
                 if idSIGN != -1 :
                     sign=body[idSIGN:]
                     idsigkey=sign.find(")")
                     idsign=sign.find("Sig_")  
                     signkey=sign[idsign+5:idsigkey] 
                     final=final+['2']+[signkey]
                  
                     idendsign=sign.rfind("}")
                     
                     
                     
                     # nested encryption of sign
                     idnestedEnc=sign.find("E_(")
                     
                     if idnestedEnc != -1:
                     
                        signature1=sign[idsigkey+3:idnestedEnc]
                        
                        idEndnestedEnc=sign.find("}")
                        signature2=sign[idEndnestedEnc+2:idendsign]
                        
                        idEnckey=sign.find(")")
                        nestedEnc=sign[idnestedEnc:]
                        idnestedEnckey=nestedEnc.find(")")
                        nestedEnc=sign[idnestedEnc:idEndnestedEnc]
                        # nested Encryption key,param
                        nestedEnckey=nestedEnc[4:idnestedEnckey]
                        nestedEncparam=nestedEnc[idnestedEnckey+3:]
                        

                        signature=signature1+','+signature2
                        final = final+signature.replace("\n", "").split(",")

                        final=final+[nestedEnckey]
                        final=final+nestedEncparam.replace("\n", "").split(",")
                        n=23-len(final)
                        final=final 
                        
                        
                        
                     else:
                        signature=sign[idsigkey+3:idendsign] 
                        final = final+signature.replace("\n", "").split(",")
                        n=23-len(final)
                        final=final 
                        
                        
                  
                     
            
                        

               # =splitting encryption

                     encr= body[idx:idSIGN]
                     idk=encr.find(")")
                     idend1=encr.find("}")
               
            
                     if idx != -1 :
                        
                        
                        idxx = encr.find("E_(")
                        KEY1=encr[idxx+3:idk]
                        final=final +['3']+ [KEY1]
                        
                  
                        X=encr[idk:idSIGN]
                        idnestsig = encr.find("sig")
                        nestsig=encr[idnestsig:idSIGN]
                        idnestsig = nestsig.find("sig")
                        idnestsigkey=nestsig.find(")")
                        nestsigkey=nestsig[idnestsig+5:idnestsigkey]
                        final = final+[nestsigkey]
                        idendsig=nestsig.find("}")
                        nestsignature=nestsig[idnestsigkey+3:idendsig]
                        final = final+nestsignature.replace("\n", "").split(",")
                        
                     
                     

                  
                        idx2=X.find("E_(")
                        
                              
                        if idx2 != -1:
                        
                           idx3=X.find("}")
                           
                           if idx2 > idx3:
                              
                              
                              if idnestsig!= -1:
                                 idnestsig = encr.find("sig")
                                 encr11=encr[idk+3:idnestsig]
                                 
                                 XX=X[idx3+1:]
                                 idend12=XX.find("}")
                                 
                                 encr12=XX[1:idend12]
                                 encr1=encr11+','+encr12
                                 final = final+encr1.replace("\n", "").split(",")
                                 idxx2=XX.find("E_(")
                                 idk2=XX.find(")")
                                 KEY2=XX[idxx2+3:idk2]
                                 final = final+['3']+[KEY2]
                                 encr2=XX[idk2+3:]
                                 idend2=encr2.find("}")
                                 encr2=encr2[:idend2]
                                 
                                 final = final+encr2.replace("\n", "").split(",")
                                 n=43-len(final)
                                 final=final +finalhash
                                 print(final)
                                 for element in final:
                                    file.write(element)
                                    file.write(',')
                                 
                                 
                                 
                              else:
                                    encr1=encr[idk+3:idend1]
                                    final = final+encr1.replace("\n", "").split(",")
                                    XX=X[idx3+1:]
                                    idxx2=XX.find("E_(")
                                    idk2=XX.find(")")
                                    KEY2=XX[idxx2+3:idk2]
                                    final = final+['3']+[KEY2]
                                    idend2=XX.find("}")
                                    encr2=XX[idk2+3:idend2]
                                    final = final+encr2.replace("\n", "").split(",")
                                    n=43-len(final)
                                    final=final +finalhash
                                    print(final)
                                    for element in final:
                                       file.write(element)
                                       file.write(',')
                                    
                                    
                              
                           
                              
                           
                           else:
                           
                              idx4=encr.rfind("}")
                              encr1=encr[idk+3:idx4]
                              idx5=encr1.find("E_(")
                              idknest=encr1.find(")")
                              KEYnest=encr1[idx5+3:idknest]
                              final = final+[KEYnest]
                              idend2=encr1.find("}")
                              encrNest=encr1[idknest+3:idend2]
                              final = final+encrNest.replace("\n", "").split(",")
                              n=43-len(final)
                              final=final +finalhash
                              print(final)
                              for element in final:
                                 file.write(element)
                                 file.write(',')

                        else:
                              encr1=encr[idk+3:idend1]
                              idnestsig = encr.find("sig")
                              encr11=encr[idk+3:idnestsig]
                              idx3=X.find("}")         
                              XX=X[idx3+1:]
                              idend12=XX.find("}")
                                       
                              encr12=XX[1:idend12]
                              encr1=encr11+','+encr12
                              final = final+encr1.replace("\n", "").split(",")
                              n=43-len(final)
                              final=final +finalhash
                              print(final)
                              for element in final:
                                 file.write(element)
                                 file.write(',')
                     else:
                        final=final +['3']+finalhash
                        print(final)
                        for element in final:
                              file.write(element)
                              file.write(',')



                           
                              

                  #when there is no signature        
                 else:
                     final=final +['2']
                     idx = body.find("E_(")
                     encr= body[idx:]
                     
               
            
                     if idx != -1 :
                        idk=encr.find(")")
                        idend1=encr.find("}")
                        idxx = encr.find("E_(")
                        KEY1=encr[idxx+3:idk]
                        final=final+['3']+[KEY1]
                              
                        X=encr[idk:idSIGN]
                        idnestsig = encr.find("sig")
                        nestsig=encr[idnestsig:idSIGN]
                        idnestsig = nestsig.find("sig")
                        idnestsigkey=nestsig.find(")")
                        nestsigkey=nestsig[idnestsig+5:idnestsigkey]
                        final = final+[nestsigkey]
                        idendsig=nestsig.find("}")
                        nestsignature=nestsig[idnestsigkey+3:idendsig]
                        final = final+nestsignature.replace("\n", "").split(",")
                        

                        
                     

                     
                        idx2=X.find("E_(")
                           
                                 
                        if idx2 != -1:
                           
                           idx3=X.find("}")
                              
                           if idx2 > idx3:
                                 
                                 
                              if idnestsig!= -1:
                                 idnestsig = encr.find("sig")
                                 encr11=encr[idk+3:idnestsig]
                                    
                                 XX=X[idx3+1:]
                                 idend12=XX.find("}")
                                    
                                 encr12=XX[1:idend12]
                                 encr1=encr11+','+encr12
                                 final = final+encr1.replace("\n", "").split(",")
                                 idxx2=XX.find("E_(")
                                 idk2=XX.find(")")
                                 KEY2=XX[idxx2+3:idk2]
                                 final = final+['3']+[KEY2]
                                 encr2=XX[idk2+3:]
                                 idend2=encr2.find("}")
                                 encr2=encr2[:idend2]
                                    
                                 final = final+encr2.replace("\n", "").split(",")
                                 n=43-len(final)
                                 final=final +finalhash
                                 print(final)
                                 for element in final:
                                       file.write(element)
                                       file.write(',')
                                 
                                 
                                    
                              else:
                                    encr1=encr[idk+3:idend1]
                                    final = final+encr1.replace("\n", "").split(",")
                                    XX=X[idx3+1:]
                                    idxx2=XX.find("E_(")
                                    idk2=XX.find(")")
                                    KEY2=XX[idxx2+3:idk2]
                                    final = final+['3']+[KEY2]
                                    idend2=XX.find("}")
                                    encr2=XX[idk2+3:idend2]
                                    final = final+encr2.replace("\n", "").split(",")
                                    n=43-len(final)
                                    final=final +finalhash
                                    print(final)
                                    for element in final:
                                       file.write(element)
                                       file.write(',')
                                    
                                    
                              
                           
                              
                              
                           else:
                              
                                 idx4=encr.rfind("}")
                                 encr1=encr[idk+3:idx4]
                                 idx5=encr1.find("E_(")
                                 idknest=encr1.find(")")
                                 KEYnest=encr1[idx5+3:idknest]
                                 final = final+[KEYnest]
                                 idend2=encr1.find("}")
                                 encrNest=encr1[idknest+3:idend2]
                                 final = final+encrNest.replace("\n", "").split(",")
                                 n=43-len(final)
                                 final=final +finalhash
                                 print(final)
                                 for element in final:
                                       file.write(element)
                                       file.write(',')
                                 

                        else:
                              encr1=encr[idk+3:idend1]
                              idnestsig = encr.find("sig")
                              encr11=encr[idk+3:idnestsig]
                              idx3=X.find("}")         
                              XX=X[idx3+1:]
                              idend12=XX.find("}")
                                          
                              encr12=XX[1:idend12]
                              encr1=encr11+','+encr12
                              final = final+encr1.replace("\n", "").split(",")
                              n=43-len(final)
                              final=final +finalhash
                              print(final)
                              for element in final:
                                 file.write(element)
                                 file.write(',')
                     else:
                              final=final +['3']+finalhash
                              print(final)
                              for element in final:
                                 file.write(element)
                                 file.write(',')
                              
                              
                              

        
       
  
    


                
                
        
      
if __name__ == "__main__":
    encode_file("Secrecy.txt")
