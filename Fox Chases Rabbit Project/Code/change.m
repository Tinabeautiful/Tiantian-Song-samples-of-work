R = 800;
s_r0 = 12;
s_f0 = 17;
mu_r = 0.0008;
mu_f = 0.0002;
OG = 300;
t_1 = 1/(mu_f*s_f0)*(exp(mu_f*OG)-1);

tspan0 = (0:0.0001:t_1)';

r0 = zeros(2,size(tspan0,1));
for i = 1:size(tspan0,1)
    r0(1,i) = rpos(tspan0(i),1,2);
    r0(2,i) = rpos(tspan0(i),2,2);
end
plot(r0(1,:),r0(2,:),'green',Linewidth=3);
hold on

x = 0;
y = linspace(0,300,10000);
plot(x,y,'-o');

TF = (exp(mu_r*R*pi/3)-1)/mu_r/s_r0;
tspan = (t_1:0.0001:TF);

r = [-R*sin(1/mu_r/R*log(mu_r*s_r0*tspan+1)); R*cos(1/mu_r/R*log(mu_r*s_r0*tspan+1))];

odefun = @(t,z) [(s_f0/(mu_f*s_f0*t+1)*(rpos(t,1,2)-z(1))/sqrt((rpos(t,1,2)-z(1))^2+(rpos(t,2,2)-z(2))^2));(s_f0/(mu_f*s_f0*t+1)*(rpos(t,2,2)-z(2))/sqrt((rpos(t,1,2)-z(1))^2+(rpos(t,2,2)-z(2))^2))];

[t,z] = ode45(odefun,tspan,[0 300]);

for i = 1:size(t,1)
    catch_distance = sqrt((r(1,i) - z(i,1))^2+(r(2,i) - z(i,2))^2);
    if catch_distance < 0.1
        disp("Caught at time t2");
        disp("Time:");
        disp([t_1 t(i)]);
        disp("Place:");
        disp([r(1,i) r(2,i)]);
        plot(z(1:i,1),z(1:i,2),'-o');
        plot(r(1,(1:i)),r(2,(1:i)),'-o');
        fox_total_distance = 1/mu_f*log(mu_f*s_f0*t(i)+1);
        disp("Fox's distance:");
        disp(fox_total_distance);
        return;
    end
    if cantsee(r(1,i),r(2,i),z(i,1),z(i,2))
        break;
    end
end
t_2 = t(i);

if ~cantsee(r(1,i),r(2,i),z(i,1),z(i,2))
    disp("Can't catch it...");
    disp("Time:");
    disp([t_1 t_2]);
    disp("Fox's position:");
    disp([z(i,1) z(i,2)]);
    return;
end
t_3 = t_2 + 1/(mu_f*s_f0)*(exp(mu_f*sqrt((350+z(mid,1))^2 + (620-z(mid,2))^2))-1);

tspan2 = t_3:0.0001:TF;
r2 = [-R*sin(1/mu_r/R*log(mu_r*s_r0*tspan2+1)); R*cos(1/mu_r/R*log(mu_r*s_r0*tspan2+1))];

[t2,z2] = ode45(odefun,tspan2,[-350 620]);

z_x = [z(i,1) z2(1,1)];
z_y = [z(i,2) z2(1,2)];
plot(z_x,z_y,LineWidth=3,Color='magenta');
plot(z2(:,1),z2(:,2),'-o');
hold off

for i = 1:size(t,1)
    catch_distance = sqrt((r2(1,i) - z2(i,1))^2+(r2(2,i) - z2(i,2))^2);
    if catch_distance < 0.1
        disp("Caught at time t4");
        disp("Time:");
        disp([t_1 t_2 t_3 t2(i)]);
        disp("Place:");
        disp([r2(1,i) r2(2,i)]);
        return;
    end
end

disp("Can't catch it...");
disp("Time:");
disp([t_1 t_2 t_3 TF]);
disp("Fox's position:");
disp([z2(i,1) z2(i,2)]);