function ReadSmoothSave()
% Function to ea and save files
c = Rload('*C001*tif',1,1); s = Rload('*C002*tif',1,1);
c_blurred = imgaussian(c.*(c>140), 3); s_blurred = imgaussian(s.*(s>140), 3);

msave(c_blurred./max(c_blurred(:)), 2); 
msave(s_blurred./max(s_blurred(:)), 3);
end