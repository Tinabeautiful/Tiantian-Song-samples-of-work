%A function that determines if the fox can see the rabbit
%Inputs consist of the positions of the rabbit and the fox
function res = cantsee(r_x,r_y,f_x,f_y)
a_x = -350;
a_y = 620;
e_x = -500;
e_y = 350;
line1 = [r_x,r_y;f_x,f_y];%The line connecting the positions of the rabbit and the fox
line2 = [a_x,a_y;e_x,e_y];%Line AE

%Return the intersection points of two polylines
[p_x,p_y] = polyxpoly(line1(:,1),line1(:,2),line2(:,1),line2(:,2));

%Determine if line1 and line2 intersect
%If they intersect, the function 'cantsee' returns true
if isempty(p_x)
    res = false;
else
    res = true;
end