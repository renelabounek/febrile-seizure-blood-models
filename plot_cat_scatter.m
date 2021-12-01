function plot_cat_scatter(Yp1,Yp2,Yp3_1,Yp3_2,Yp4,YLBL,loc,seiz_risk)
%PLOT_CAT_SCATTER Summary of this function goes here
%   Detailed explanation goes here
    Yp3= [Yp3_1; Yp3_2];
    Yp1_mean = median(Yp1);
    Yp2_mean = median(Yp2);
    Yp3_mean = median(Yp3);
    Yp4_mean = median(Yp4);

    Yp1_Cinterval = quantile(Yp1,[0.025 0.25 0.50 0.75 0.975]);
    Yp2_Cinterval = quantile(Yp2,[0.025 0.25 0.50 0.75 0.975]);
    Yp3_Cinterval = quantile(Yp3,[0.025 0.25 0.50 0.75 0.975]);
    Yp4_Cinterval = quantile(Yp4,[0.025 0.25 0.50 0.75 0.975]);

    plot([0.27 0.73],[Yp1_Cinterval(2) Yp1_Cinterval(2)],'k','LineWidth',3);
    hold on
    if ~isempty(seiz_risk)
            H3=plot([0 5],[seiz_risk(1,1) seiz_risk(1,1)],':','LineWidth',4,'Color',[0.5 0.5 0.5]);
    end
    plot([0.27 0.73],[Yp1_Cinterval(4) Yp1_Cinterval(4)],'k','LineWidth',3);
    plot([0.27 0.27],[Yp1_Cinterval(2) Yp1_Cinterval(4)],'k','LineWidth',3);
    plot([0.73 0.73],[Yp1_Cinterval(2) Yp1_Cinterval(4)],'k','LineWidth',3);
    plot([0.5 0.5],[Yp1_Cinterval(1) Yp1_Cinterval(2)],'k','LineWidth',3);
    plot([0.48 0.52],[Yp1_Cinterval(1) Yp1_Cinterval(1)],'k','LineWidth',3);
    plot([0.5 0.5],[Yp1_Cinterval(4) Yp1_Cinterval(5)],'k','LineWidth',3);
    plot([0.48 0.52],[Yp1_Cinterval(5) Yp1_Cinterval(5)],'k','LineWidth',3);

    plot([0.77 1.23],[Yp2_Cinterval(2) Yp2_Cinterval(2)],'k','LineWidth',3);
    plot([0.77 1.23],[Yp2_Cinterval(4) Yp2_Cinterval(4)],'k','LineWidth',3);
    plot([0.77 0.77],[Yp2_Cinterval(2) Yp2_Cinterval(4)],'k','LineWidth',3);
    plot([1.23 1.23],[Yp2_Cinterval(2) Yp2_Cinterval(4)],'k','LineWidth',3);
    plot([1.0 1.0],[Yp2_Cinterval(1) Yp2_Cinterval(2)],'k','LineWidth',3);
    plot([0.98 1.02],[Yp2_Cinterval(1) Yp2_Cinterval(1)],'k','LineWidth',3);
    plot([1.0 1.0],[Yp2_Cinterval(4) Yp2_Cinterval(5)],'k','LineWidth',3);
    plot([0.98 1.02],[Yp2_Cinterval(5) Yp2_Cinterval(5)],'k','LineWidth',3);

    plot([1.27 1.73],[Yp3_Cinterval(2) Yp3_Cinterval(2)],'k','LineWidth',3);
    plot([1.27 1.73],[Yp3_Cinterval(4) Yp3_Cinterval(4)],'k','LineWidth',3);
    plot([1.27 1.27],[Yp3_Cinterval(2) Yp3_Cinterval(4)],'k','LineWidth',3);
    plot([1.73 1.73],[Yp3_Cinterval(2) Yp3_Cinterval(4)],'k','LineWidth',3);
    plot([1.5 1.5],[Yp3_Cinterval(1) Yp3_Cinterval(2)],'k','LineWidth',3);
    plot([1.48 1.52],[Yp3_Cinterval(1) Yp3_Cinterval(1)],'k','LineWidth',3);
    plot([1.5 1.5],[Yp3_Cinterval(4) Yp3_Cinterval(5)],'k','LineWidth',3);
    plot([1.48 1.52],[Yp3_Cinterval(5) Yp3_Cinterval(5)],'k','LineWidth',3);

    plot([1.77 2.23],[Yp4_Cinterval(2) Yp4_Cinterval(2)],'k','LineWidth',3);
    plot([1.77 2.23],[Yp4_Cinterval(4) Yp4_Cinterval(4)],'k','LineWidth',3);
    plot([1.77 1.77],[Yp4_Cinterval(2) Yp4_Cinterval(4)],'k','LineWidth',3);
    plot([2.23 2.23],[Yp4_Cinterval(2) Yp4_Cinterval(4)],'k','LineWidth',3);
    plot([2.0 2.0],[Yp4_Cinterval(1) Yp4_Cinterval(2)],'k','LineWidth',3);
    plot([1.98 2.02],[Yp4_Cinterval(1) Yp4_Cinterval(1)],'k','LineWidth',3);
    plot([2.0 2.0],[Yp4_Cinterval(4) Yp4_Cinterval(5)],'k','LineWidth',3);
    plot([1.98 2.02],[Yp4_Cinterval(5) Yp4_Cinterval(5)],'k','LineWidth',3);

    scatter(1*ones(size(Yp1))/2,Yp1,850, 'y.', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7);
    scatter(2*ones(size(Yp2))/2,Yp2,850, 'g.', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7);
    scatter(3*ones(size(Yp3_1))/2,Yp3_1,850, 'b.', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7);
    scatter(3*ones(size(Yp3_2))/2,Yp3_2,100, 'b*', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7,'LineWidth',2);
    scatter(4*ones(size(Yp4))/2,Yp4,850, 'r.', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7);
    H1 = plot([0.27 0.73],[Yp1_mean Yp1_mean],'k-.','LineWidth',3);
    H2 = scatter(-5,Yp3_mean,100,'b*','LineWidth',3);
    plot([0.77 1.23],[Yp2_mean Yp2_mean],'k-.','LineWidth',3);
    plot([1.27 1.73],[Yp3_mean Yp3_mean],'k-.','LineWidth',3);
    plot([1.77 2.23],[Yp4_mean Yp4_mean],'k-.','LineWidth',3);
    hold off
    grid on
    xlim([0.2 2.3])
    if ~strcmp(loc,' ') && isempty(seiz_risk)
            legend([H1, H2],{'median value','complex cases'},'location',loc)
    elseif ~strcmp(loc,' ') && ~isempty(seiz_risk)
            legend([H1, H2, H3],{'median value','complex cases',['thr=' num2str(seiz_risk(1,1),'%10.2f') ';SE=' num2str(seiz_risk(1,2),'%10.1f') '%;SP=' num2str(seiz_risk(1,3),'%10.1f') '%']},'location',loc)
    elseif strcmp(loc,' ') && ~isempty(seiz_risk)
            legend(H3,{['thr=' num2str(seiz_risk(1,1),'%10.2f') ';SE=' num2str(seiz_risk(1,2),'%10.1f') '%;SP=' num2str(seiz_risk(1,3),'%10.1f') '%']},'location','northeast')
    end
    ylabel(YLBL)
    set(gca,'XTick',(1:1:4)/2,...
             'XTickLabel',{'healthy controls (1)'
                           'without seizures (2)' 
                           'non-recurrent seizures (3)'
                           'recurrent seizures (4)'
                           },...
             'TickLength',[0 0],'LineWidth',2,...
             'FontSize',14)
    xtickangle(16)
end

