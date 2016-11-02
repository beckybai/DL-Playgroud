function pic  = pic(a)
 a = a - min(min(a));
 a = a.*255/(max(max(a))-min(min(a)));
 a = int8(a);
 imshow(a);
end