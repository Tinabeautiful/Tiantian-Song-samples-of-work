%A function that determines the position of the rabbit
%t=time; axis: 1->x-axis, 2->y-axis; stage=question number (1 or 2)
function res = rpos(t,axis,stage)
R = 800;
s0 = 12;
if stage == 1
    if axis == 1
        res = -R*sin(s0*t/R);
    else
        res = R*cos(s0*t/R);
    end
else
    mu_r = 0.0008;
    if axis == 1
        res = -R*sin(1/mu_r/R*log(mu_r*s0*t+1));
    else
        res = R*cos(1/mu_r/R*log(mu_r*s0*t+1));
    end
end